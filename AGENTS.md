# Distributed Private Key Security

## Project Purpose

This project implements distributed private key management using threshold cryptography.

Goal:

- Eliminate single points of failure
- Prevent exposure of complete private keys
- Support secure key reconstruction
- Support distributed authorization
- Maintain confidentiality of key material

The system should be analyzed as a cryptographic protocol, not merely as application code.

---

## Development Workflow

Before implementing any feature:

1. Understand the protocol flow
2. Identify trust assumptions
3. Identify attack surfaces
4. Analyze security implications
5. Implement
6. Write tests
7. Review for cryptographic correctness

Never implement cryptographic features without explaining security assumptions.

---

## Security Principles

Always assume:

- Malicious clients
- Network adversaries
- Insider threats
- Node compromise
- Replay attempts
- Message tampering

Evaluate:

- Confidentiality
- Integrity
- Availability
- Authentication
- Authorization

---

## Cryptography Rules

Never:

- Invent cryptographic primitives
- Modify standard algorithms
- Create custom encryption schemes
- Store plaintext private keys
- Log private keys or secret shares

Always:

- Use audited libraries
- Use cryptographically secure randomness
- Validate inputs
- Verify signatures
- Handle edge cases

Preferred Standards:

- AES-GCM
- ChaCha20-Poly1305
- HKDF
- Ed25519
- X25519
- secp256k1
- SHA-256
- SHA-3
- BLAKE3

---

## Threshold Cryptography

Assume threshold cryptography is a core component.

When analyzing threshold systems:

Explain:

- Threshold parameters (t,n)
- Reconstruction requirements
- Fault tolerance
- Share compromise impact
- Collusion assumptions

Always identify:

- Minimum shares required
- Maximum tolerable compromised nodes
- Reconstruction attack vectors

Shamir Secret Sharing implementations should be reviewed for correctness and boundary conditions. Threshold key management systems commonly rely on secret sharing or threshold schemes to avoid single points of failure. :contentReference[oaicite:0]{index=0}

---

## Distributed Key Generation

When DKG is used:

Analyze:

- Trusted dealer assumptions
- Share verification
- Public verification
- Key refresh procedures
- Participant honesty assumptions

Always explain:

- Setup phase
- Distribution phase
- Reconstruction phase

Distributed Key Generation removes reliance on a single trusted key creator and is a foundational primitive in threshold systems. :contentReference[oaicite:1]{index=1}

---

## Threat Modeling

For every security-sensitive change provide:

### Assets

What is being protected?

Examples:

- Private keys
- Secret shares
- Session keys
- Recovery data

### Adversaries

Examples:

- External attacker
- Malicious participant
- Compromised node
- Rogue operator

### Security Goals

Examples:

- Key confidentiality
- Share confidentiality
- Availability
- Non-repudiation

### Attack Scenarios

Analyze:

- Share theft
- Replay attacks
- Rogue participant attacks
- Key reconstruction attacks
- Node compromise
- Insider collusion

---

## Code Quality

Always:

- Use TypeScript strict mode
- Use explicit types
- Validate inputs
- Add error handling
- Add logging
- Add unit tests

Never:

- Use any
- Ignore exceptions
- Leave TODO implementations

---

## Testing Requirements

Every security-sensitive change must include:

1. Unit tests
2. Failure-path tests
3. Invalid-input tests
4. Reconstruction tests
5. Threshold-boundary tests

Examples:

- t-1 shares fail reconstruction
- t shares succeed
- Invalid share rejected
- Tampered share rejected

---

## Review Checklist

Before completion:

- Does this leak key material?
- Does this weaken threshold guarantees?
- Does this introduce a trusted party?
- Does this create replay opportunities?
- Does this break fault tolerance?
- Does this reduce cryptographic security assumptions?

Provide a final security assessment for all cryptographic changes.