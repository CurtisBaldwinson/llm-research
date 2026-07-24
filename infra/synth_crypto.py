#!/usr/bin/env python3
"""synth_crypto: sovereign-key hybrid PQ encryption for remote run results.

Threat model and design
=======================

Synthetic experiment results are encrypted on the pod with a single
*public* recipient key and decrypted on the operator's machine with the
matching *private* key.  The pod never holds a private key; even a
fully compromised pod can encrypt new bogus payloads but cannot decrypt
past results.

We use a hybrid KEM (X25519 + ML-KEM-768) to combine:

  * classical post-IETF-RFC X25519 (32-byte shared secret, conservative,
    battle-tested), and
  * post-quantum ML-KEM-768 (NIST FIPS-203 standardized August 2024,
    NIST Level 3 security claim).

An adversary must break BOTH primitives to recover the data encryption
key.  Concatenation-combiner via HKDF-SHA256 is IND-CCA2 secure as long
as at least one of the underlying KEMs remains IND-CCA2 secure.

Bulk encryption uses ChaCha20-Poly1305 (an AEAD construction; 256-bit
key, 96-bit nonce, 128-bit auth tag).  The full header (magic, version,
ephemeral keys, KEM ciphertext, nonce) is bound as Associated Data so
any tampering is detected.

Wire format (synth-crypto-v2)
=============================

  Bytes      Content
  ---------- -----------------------------------------------------
  0..5       Magic "SYNTH\\x00"  (6 bytes, ASCII)
  6          Format version = 0x02                  (uint8)
  7          Reserved        = 0x00                 (uint8)
  8..39      X25519 ephemeral public key            (32 bytes)
  40..1127   ML-KEM-768 KEM ciphertext              (1088 bytes)
  1128..1139 ChaCha20-Poly1305 nonce                (12 bytes)
  1140..N    AEAD ciphertext (= plaintext + 16-byte tag)

  Header length: 1140 bytes
  Per-file overhead: 1140 + 16 = 1156 bytes

Key derivation
==============

  x25519_ss = X25519.dh(ephemeral_priv, recipient_x25519_pub)    # 32 B
  mlkem_ss  = ML-KEM-768.encap(recipient_mlkem_pub) -> (ct, ss)  # 32 B
  ikm       = x25519_ss || mlkem_ss                              # 64 B
  dek       = HKDF-SHA256(ikm, salt=SALT, info=INFO, length=32)

Public / private key file format
================================

  # Synthetic crypto recipient public key (v2: X25519 + ML-KEM-768 hybrid)
  # Generated: <utc>
  # Comment: <free text>

  synth-crypto-v2 <base64(x25519_pub || mlkem_pub)>

Private key file is symmetric but with header label
``synth-crypto-v2-secret`` and the encoded blob is
``base64(x25519_priv || mlkem_priv)``.

Wire-format versioning
======================

Byte 6 of the wire format is reserved for future algorithm rotations.
``decrypt`` rejects any version other than 0x02; downgrade attacks are
not possible.  A future v3 (e.g., ML-KEM-1024 or a different bulk
cipher) introduces a new version byte and a fresh key-file label.

Dependencies
============

  * ``cryptography`` (X25519, HKDF-SHA256, ChaCha20-Poly1305)
  * ``pqcrypto`` (ML-KEM-768; ships pre-built wheels with the NIST
    reference implementation — no system library setup required, so
    pod bootstrap is just ``pip install pqcrypto``).
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Wire-format constants
# ---------------------------------------------------------------------------

MAGIC = b"SYNTH\x00"
WIRE_VERSION = 0x02
RESERVED_BYTE = 0x00

X25519_KEY_LEN = 32           # NIST/Bernstein/IETF Curve25519
MLKEM_PUB_LEN = 1184          # FIPS-203, ML-KEM-768
MLKEM_PRIV_LEN = 2400         # FIPS-203, ML-KEM-768
MLKEM_CT_LEN = 1088           # FIPS-203, ML-KEM-768
KEM_SS_LEN = 32               # both KEMs emit 32-byte shared secrets
DEK_LEN = 32                  # ChaCha20-Poly1305 key length
NONCE_LEN = 12                # ChaCha20-Poly1305 nonce length

HEADER_LEN = (
    len(MAGIC) + 1 + 1 + X25519_KEY_LEN + MLKEM_CT_LEN + NONCE_LEN
)  # 1140

HKDF_SALT = b"synth-crypto-v2-salt"
HKDF_INFO = b"synth-crypto-v2-dek"

PUBKEY_LABEL = "synth-crypto-v2"
PRIVKEY_LABEL = "synth-crypto-v2-secret"
MLKEM_ALG = "ML-KEM-768"


# ---------------------------------------------------------------------------
# Key types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PublicKey:
    """Hybrid X25519 + ML-KEM-768 recipient public key."""

    x25519_pub: bytes  # 32 bytes
    mlkem_pub: bytes   # 1184 bytes

    def __post_init__(self) -> None:
        if len(self.x25519_pub) != X25519_KEY_LEN:
            raise ValueError(
                f"x25519_pub must be {X25519_KEY_LEN} bytes, "
                f"got {len(self.x25519_pub)}"
            )
        if len(self.mlkem_pub) != MLKEM_PUB_LEN:
            raise ValueError(
                f"mlkem_pub must be {MLKEM_PUB_LEN} bytes, "
                f"got {len(self.mlkem_pub)}"
            )

    def to_text(self, comment: Optional[str] = None) -> str:
        blob = base64.b64encode(self.x25519_pub + self.mlkem_pub).decode("ascii")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        comment_line = f"# Comment: {comment}\n" if comment else ""
        return (
            f"# Synthetic crypto recipient public key (v2: X25519 + ML-KEM-768 hybrid)\n"
            f"# Generated: {now}\n"
            f"{comment_line}"
            f"\n"
            f"{PUBKEY_LABEL} {blob}\n"
        )

    @classmethod
    def from_text(cls, text: str) -> "PublicKey":
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2 or parts[0] != PUBKEY_LABEL:
                raise ValueError(
                    f"unrecognized public-key line; expected label "
                    f"{PUBKEY_LABEL!r}, got {parts[0]!r}"
                )
            blob = base64.b64decode(parts[1])
            if len(blob) != X25519_KEY_LEN + MLKEM_PUB_LEN:
                raise ValueError(
                    f"public-key blob length {len(blob)} != "
                    f"{X25519_KEY_LEN + MLKEM_PUB_LEN}"
                )
            return cls(
                x25519_pub=blob[:X25519_KEY_LEN],
                mlkem_pub=blob[X25519_KEY_LEN:],
            )
        raise ValueError("no public-key line found in input")

    @classmethod
    def from_file(cls, path: Path) -> "PublicKey":
        return cls.from_text(Path(path).read_text())


@dataclasses.dataclass(frozen=True)
class PrivateKey:
    """Hybrid X25519 + ML-KEM-768 recipient private key.

    Holds both private halves.  Decryption requires both: an adversary
    with only one half cannot recover the data encryption key.
    """

    x25519_priv: bytes  # 32 bytes
    mlkem_priv: bytes   # 2400 bytes

    def __post_init__(self) -> None:
        if len(self.x25519_priv) != X25519_KEY_LEN:
            raise ValueError(
                f"x25519_priv must be {X25519_KEY_LEN} bytes, "
                f"got {len(self.x25519_priv)}"
            )
        if len(self.mlkem_priv) != MLKEM_PRIV_LEN:
            raise ValueError(
                f"mlkem_priv must be {MLKEM_PRIV_LEN} bytes, "
                f"got {len(self.mlkem_priv)}"
            )

    def to_text(self) -> str:
        blob = base64.b64encode(self.x25519_priv + self.mlkem_priv).decode("ascii")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return (
            f"# Synthetic crypto recipient PRIVATE key (v2)\n"
            f"# Generated: {now}\n"
            f"# !!! KEEP SECRET — never commit to a repository !!!\n"
            f"\n"
            f"{PRIVKEY_LABEL} {blob}\n"
        )

    @classmethod
    def from_text(cls, text: str) -> "PrivateKey":
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2 or parts[0] != PRIVKEY_LABEL:
                raise ValueError(
                    f"unrecognized private-key line; expected label "
                    f"{PRIVKEY_LABEL!r}, got {parts[0]!r}"
                )
            blob = base64.b64decode(parts[1])
            if len(blob) != X25519_KEY_LEN + MLKEM_PRIV_LEN:
                raise ValueError(
                    f"private-key blob length {len(blob)} != "
                    f"{X25519_KEY_LEN + MLKEM_PRIV_LEN}"
                )
            return cls(
                x25519_priv=blob[:X25519_KEY_LEN],
                mlkem_priv=blob[X25519_KEY_LEN:],
            )
        raise ValueError("no private-key line found in input")

    @classmethod
    def from_file(cls, path: Path) -> "PrivateKey":
        return cls.from_text(Path(path).read_text())


# ---------------------------------------------------------------------------
# Cryptographic primitives (lazy imports keep the module importable even
# when the optional deps aren't installed, for CLI --help and key parsing).
# ---------------------------------------------------------------------------


def _import_classical():
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import x25519
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return hashes, serialization, x25519, ChaCha20Poly1305, HKDF


def _import_mlkem():
    try:
        from pqcrypto.kem import ml_kem_768  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pqcrypto is required for ML-KEM-768 support. "
            "Install via `pip install pqcrypto` (ships pre-built "
            "wheels for macOS and Linux; no system library needed)."
        ) from e
    return ml_kem_768


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------


def generate_keypair() -> Tuple[PublicKey, PrivateKey]:
    """Generate a fresh hybrid keypair using OS randomness."""
    _, serialization, x25519, _, _ = _import_classical()
    ml_kem_768 = _import_mlkem()

    x_priv_obj = x25519.X25519PrivateKey.generate()
    x_priv = x_priv_obj.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    x_pub = x_priv_obj.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    m_pub, m_priv = ml_kem_768.generate_keypair()

    return PublicKey(x25519_pub=x_pub, mlkem_pub=m_pub), PrivateKey(
        x25519_priv=x_priv, mlkem_priv=m_priv
    )


# ---------------------------------------------------------------------------
# Encryption / decryption
# ---------------------------------------------------------------------------


def _derive_dek(x25519_ss: bytes, mlkem_ss: bytes) -> bytes:
    hashes, _, _, _, HKDF = _import_classical()
    ikm = x25519_ss + mlkem_ss
    return HKDF(
        algorithm=hashes.SHA256(),
        length=DEK_LEN,
        salt=HKDF_SALT,
        info=HKDF_INFO,
    ).derive(ikm)


def encrypt(plaintext: bytes, recipient: PublicKey) -> bytes:
    """Encrypt ``plaintext`` for ``recipient``; return wire-format ciphertext.

    Output layout: see module docstring.  Per-file overhead is fixed at
    1156 bytes regardless of plaintext size.
    """
    _, serialization, x25519, ChaCha20Poly1305, _ = _import_classical()
    ml_kem_768 = _import_mlkem()

    # Classical leg: ephemeral X25519 + DH with recipient X25519 pubkey.
    ephem_priv = x25519.X25519PrivateKey.generate()
    ephem_pub_bytes = ephem_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    recipient_x_pub = x25519.X25519PublicKey.from_public_bytes(recipient.x25519_pub)
    x_ss = ephem_priv.exchange(recipient_x_pub)
    assert len(x_ss) == KEM_SS_LEN

    # PQ leg: ML-KEM-768 encapsulation against recipient ML-KEM pubkey.
    # `pqcrypto.kem.ml_kem_768.encrypt(pk)` performs KEM encapsulation
    # (returns ciphertext + shared secret); the naming follows pqcrypto's
    # convention, not the IETF KEM API naming.
    kem_ct, m_ss = ml_kem_768.encrypt(recipient.mlkem_pub)
    assert len(kem_ct) == MLKEM_CT_LEN
    assert len(m_ss) == KEM_SS_LEN

    dek = _derive_dek(x_ss, m_ss)
    nonce = os.urandom(NONCE_LEN)

    header = (
        MAGIC
        + bytes([WIRE_VERSION, RESERVED_BYTE])
        + ephem_pub_bytes
        + kem_ct
        + nonce
    )
    assert len(header) == HEADER_LEN, (len(header), HEADER_LEN)

    aead = ChaCha20Poly1305(dek)
    # Bind the entire header as Associated Data so tampering with the
    # version byte, ephemeral pubkey, or KEM ciphertext fails the tag check.
    ct = aead.encrypt(nonce, plaintext, header)
    return header + ct


def decrypt(blob: bytes, identity: PrivateKey) -> bytes:
    """Decrypt a wire-format blob with the matching private key.

    Raises ``ValueError`` on any of: wrong magic, unknown version,
    truncated blob, wrong recipient (KEM decap mismatch), AEAD tag
    failure (tampering).
    """
    _, _, x25519, ChaCha20Poly1305, _ = _import_classical()
    ml_kem_768 = _import_mlkem()

    if len(blob) < HEADER_LEN + 16:  # tag is 16 bytes
        raise ValueError(
            f"blob too short ({len(blob)} bytes); need at least {HEADER_LEN + 16}"
        )
    if blob[: len(MAGIC)] != MAGIC:
        raise ValueError("bad magic; not a synth-crypto blob")
    if blob[len(MAGIC)] != WIRE_VERSION:
        raise ValueError(
            f"unsupported wire version 0x{blob[len(MAGIC)]:02x}; "
            f"this build only accepts 0x{WIRE_VERSION:02x}"
        )
    if blob[len(MAGIC) + 1] != RESERVED_BYTE:
        raise ValueError(
            f"reserved byte must be 0x{RESERVED_BYTE:02x}; got "
            f"0x{blob[len(MAGIC) + 1]:02x}"
        )

    off = len(MAGIC) + 2
    ephem_pub_bytes = blob[off : off + X25519_KEY_LEN]
    off += X25519_KEY_LEN
    kem_ct = blob[off : off + MLKEM_CT_LEN]
    off += MLKEM_CT_LEN
    nonce = blob[off : off + NONCE_LEN]
    off += NONCE_LEN
    assert off == HEADER_LEN
    header = blob[:HEADER_LEN]
    ct = blob[HEADER_LEN:]

    # Classical leg.
    recipient_x_priv = x25519.X25519PrivateKey.from_private_bytes(identity.x25519_priv)
    ephem_pub = x25519.X25519PublicKey.from_public_bytes(ephem_pub_bytes)
    x_ss = recipient_x_priv.exchange(ephem_pub)

    # PQ leg: KEM decapsulation.  `pqcrypto.kem.ml_kem_768.decrypt`
    # returns the shared secret (same semantics as IETF KEM Decaps).
    m_ss = ml_kem_768.decrypt(identity.mlkem_priv, kem_ct)

    dek = _derive_dek(x_ss, m_ss)
    aead = ChaCha20Poly1305(dek)
    return aead.decrypt(nonce, ct, header)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_keygen(args: argparse.Namespace) -> int:
    pub, priv = generate_keypair()
    pub_path = Path(args.pub).expanduser()
    priv_path = Path(args.priv).expanduser()
    pub_path.parent.mkdir(parents=True, exist_ok=True)
    priv_path.parent.mkdir(parents=True, exist_ok=True)

    pub_path.write_text(pub.to_text(comment=args.comment))
    priv_path.write_text(priv.to_text())
    # chmod 600 on the private key file (best-effort; not a substitute
    # for proper at-rest protection, but blocks casual world-read).
    try:
        os.chmod(priv_path, 0o600)
    except OSError:
        pass
    print(f"public key  -> {pub_path}")
    print(f"private key -> {priv_path}  (chmod 600)")
    return 0


def _cmd_encrypt(args: argparse.Namespace) -> int:
    pub = PublicKey.from_file(Path(args.recipient).expanduser())
    inp = sys.stdin.buffer.read() if args.input == "-" else Path(args.input).read_bytes()
    out = encrypt(inp, pub)
    if args.output == "-":
        sys.stdout.buffer.write(out)
    else:
        Path(args.output).write_bytes(out)
    return 0


def _cmd_decrypt(args: argparse.Namespace) -> int:
    priv = PrivateKey.from_file(Path(args.identity).expanduser())
    inp = sys.stdin.buffer.read() if args.input == "-" else Path(args.input).read_bytes()
    out = decrypt(inp, priv)
    if args.output == "-":
        sys.stdout.buffer.write(out)
    else:
        Path(args.output).write_bytes(out)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="synth_crypto",
        description="Sovereign-key hybrid PQ encryption for remote run results.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_kg = sub.add_parser("keygen", help="generate a fresh hybrid keypair")
    p_kg.add_argument("--pub", required=True, help="output path for public key")
    p_kg.add_argument("--priv", required=True, help="output path for private key")
    p_kg.add_argument("--comment", help="optional free-text comment for the public key file")
    p_kg.set_defaults(func=_cmd_keygen)

    p_enc = sub.add_parser("encrypt", help="encrypt a file for a recipient")
    p_enc.add_argument("--recipient", required=True, help="path to recipient public key")
    p_enc.add_argument("--input", default="-", help="plaintext input path (default: stdin)")
    p_enc.add_argument("--output", default="-", help="ciphertext output path (default: stdout)")
    p_enc.set_defaults(func=_cmd_encrypt)

    p_dec = sub.add_parser("decrypt", help="decrypt a file with a private key")
    p_dec.add_argument("--identity", required=True, help="path to recipient private key")
    p_dec.add_argument("--input", default="-", help="ciphertext input path (default: stdin)")
    p_dec.add_argument("--output", default="-", help="plaintext output path (default: stdout)")
    p_dec.set_defaults(func=_cmd_decrypt)

    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
