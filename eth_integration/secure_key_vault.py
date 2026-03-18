"""
SecureKeyVault — Secure private key storage and retrieval.

Provides a pluggable abstraction for managing Ethereum private keys without
storing them in plaintext in .env files or the codebase. Supports multiple
backends:

    1. EncryptedFileVault (default) — AES-256-GCM encrypted keystore file
    2. EnvironmentVault — reads from env vars (for CI/testing only)
    3. Custom backends — implement the KeyVault protocol

Usage:
    from eth_integration.secure_key_vault import SecureKeyVault

    # Interactive: prompts for password to decrypt the keystore
    vault = SecureKeyVault.from_encrypted_file("vault_keys/deployer.enc")
    private_key = vault.get_key("deployer")

    # Or initialize the vault and store a key (first-time setup):
    vault = SecureKeyVault.create_encrypted_file(
        "vault_keys/deployer.enc", password="strong-passphrase"
    )
    vault.store_key("deployer", "your_hex_private_key_here")

Security:
    - Keys are never written to disk in plaintext
    - AES-256-GCM with scrypt-derived encryption keys
    - Vault files (.enc) should be added to .gitignore (already done)
    - Password can be supplied via VAULT_PASSWORD env var for automation
"""

import json
import os
import getpass
import hashlib
import secrets as stdlib_secrets
from abc import ABC, abstractmethod
from typing import Optional

from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt


class KeyVault(ABC):
    """Abstract base class for key vault backends."""

    @abstractmethod
    def get_key(self, key_name: str) -> str:
        """
        Retrieve a private key by name.

        Args:
            key_name: Identifier for the key (e.g. "deployer")

        Returns:
            Hex-encoded private key (without 0x prefix)

        Raises:
            KeyError: If key_name not found in vault
            RuntimeError: If vault cannot be accessed
        """
        ...

    @abstractmethod
    def store_key(self, key_name: str, private_key: str) -> None:
        """
        Store a private key in the vault.

        Args:
            key_name: Identifier for the key
            private_key: Hex-encoded private key (with or without 0x prefix)
        """
        ...

    @abstractmethod
    def has_key(self, key_name: str) -> bool:
        """Check if a key exists in the vault."""
        ...

    @abstractmethod
    def list_keys(self) -> list[str]:
        """List all key names in the vault."""
        ...


class EncryptedFileVault(KeyVault):
    """
    AES-256-GCM encrypted file-based key vault.

    Uses scrypt for password-based key derivation and AES-256-GCM for
    authenticated encryption of the key store.

    File format:
        16-byte salt | 16-byte nonce | 16-byte tag | ciphertext

    The ciphertext is a JSON dict mapping key_name → hex_private_key.
    """

    SCRYPT_N = 2**17  # CPU/memory cost
    SCRYPT_R = 8
    SCRYPT_P = 1
    KEY_LEN = 32  # AES-256

    def __init__(self, filepath: str, password: str):
        self._filepath = filepath
        self._password = password
        self._keys: dict[str, str] = {}

        if os.path.exists(filepath):
            self._load()

    def _derive_encryption_key(self, salt: bytes) -> bytes:
        """Derive AES-256 key from password using scrypt."""
        return scrypt(
            self._password.encode("utf-8"),
            salt,
            self.KEY_LEN,
            self.SCRYPT_N,
            self.SCRYPT_R,
            self.SCRYPT_P,
        )

    def _load(self) -> None:
        """Load and decrypt the key store from file."""
        with open(self._filepath, "rb") as f:
            data = f.read()

        if len(data) < 48:  # salt(16) + nonce(16) + tag(16)
            raise RuntimeError(f"Vault file {self._filepath} is corrupted")

        salt = data[:16]
        nonce = data[16:32]
        tag = data[32:48]
        ciphertext = data[48:]

        enc_key = self._derive_encryption_key(salt)
        cipher = AES.new(enc_key, AES.MODE_GCM, nonce=nonce)

        try:
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        except ValueError:
            raise RuntimeError(
                "Failed to decrypt vault — wrong password or corrupted file"
            )

        self._keys = json.loads(plaintext.decode("utf-8"))

    def _save(self) -> None:
        """Encrypt and save the key store to file."""
        os.makedirs(os.path.dirname(self._filepath) or ".", exist_ok=True)

        salt = stdlib_secrets.token_bytes(16)
        enc_key = self._derive_encryption_key(salt)

        cipher = AES.new(enc_key, AES.MODE_GCM)
        plaintext = json.dumps(self._keys).encode("utf-8")
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)

        with open(self._filepath, "wb") as f:
            f.write(salt + cipher.nonce + tag + ciphertext)

        # Set restrictive permissions
        os.chmod(self._filepath, 0o600)

    def get_key(self, key_name: str) -> str:
        if key_name not in self._keys:
            raise KeyError(f"Key '{key_name}' not found in vault")
        return self._keys[key_name]

    def store_key(self, key_name: str, private_key: str) -> None:
        # Normalize: strip 0x prefix
        if private_key.startswith("0x") or private_key.startswith("0X"):
            private_key = private_key[2:]
        self._keys[key_name] = private_key
        self._save()

    def has_key(self, key_name: str) -> bool:
        return key_name in self._keys

    def list_keys(self) -> list[str]:
        return list(self._keys.keys())


class EnvironmentVault(KeyVault):
    """
    Environment variable-based key vault.

    Reads keys from environment variables with the pattern:
        VAULT_KEY_{KEY_NAME_UPPER}

    Intended for CI/CD and automated testing only.
    NOT recommended for production use.
    """

    ENV_PREFIX = "VAULT_KEY_"

    def get_key(self, key_name: str) -> str:
        env_var = f"{self.ENV_PREFIX}{key_name.upper()}"
        value = os.getenv(env_var)
        if not value:
            raise KeyError(
                f"Key '{key_name}' not found (set {env_var} env var)"
            )
        if value.startswith("0x") or value.startswith("0X"):
            value = value[2:]
        return value

    def store_key(self, key_name: str, private_key: str) -> None:
        raise RuntimeError("EnvironmentVault is read-only")

    def has_key(self, key_name: str) -> bool:
        env_var = f"{self.ENV_PREFIX}{key_name.upper()}"
        return os.getenv(env_var) is not None

    def list_keys(self) -> list[str]:
        return [
            k[len(self.ENV_PREFIX):].lower()
            for k in os.environ
            if k.startswith(self.ENV_PREFIX)
        ]


class SecureKeyVault:
    """
    Factory and convenience wrapper for key vault backends.

    Provides static methods to create vault instances and a unified
    interface for the rest of the codebase.
    """

    @staticmethod
    def from_encrypted_file(
        filepath: str = "vault_keys/deployer.enc",
        password: Optional[str] = None,
    ) -> EncryptedFileVault:
        """
        Open an encrypted file vault.

        Args:
            filepath: Path to the .enc vault file
            password: Decryption password. If None, checks VAULT_PASSWORD
                     env var, then prompts interactively.

        Returns:
            EncryptedFileVault instance
        """
        if password is None:
            password = os.getenv("VAULT_PASSWORD")
        if password is None:
            password = getpass.getpass(f"Vault password for {filepath}: ")
        return EncryptedFileVault(filepath, password)

    @staticmethod
    def create_encrypted_file(
        filepath: str = "vault_keys/deployer.enc",
        password: Optional[str] = None,
    ) -> EncryptedFileVault:
        """
        Create a new encrypted file vault.

        Args:
            filepath: Path for the new .enc vault file
            password: Encryption password. If None, prompts interactively.

        Returns:
            EncryptedFileVault instance (empty, ready for store_key calls)
        """
        if password is None:
            password = getpass.getpass(f"Choose vault password for {filepath}: ")
        return EncryptedFileVault(filepath, password)

    @staticmethod
    def from_environment() -> EnvironmentVault:
        """
        Create an environment variable-based vault.

        Keys are read from VAULT_KEY_{NAME} environment variables.
        Suitable for CI/CD only.
        """
        return EnvironmentVault()
