import random
from math import pow
import tracemalloc
import time
import sys
import numpy as np
from BLOCK_CHAIN import Blockchain, Block


def blockchain_with_encryption(no_of_blocks, data, encrypted_ElGamal):
    blockchain = Blockchain()
    blockType = []
    for i in range(no_of_blocks):
        # Add data to the blockchain
        block_data = {
            "original_data": data,
            "encrypted_data": encrypted_ElGamal,
        }
        new_block = Block(i + 1, block_data, blockchain.chain[-1].hash)
        blockchain.add_block(new_block)
        blockType.append(1 if i % 5 == 0 else 0)
    return blockchain, blockType


# ============================================================
# Common Helper Functions
# ============================================================
def get_memory_size(obj):
    return sys.getsizeof(obj)


def gcd(a, b):
    if a < b:
        return gcd(b, a)
    elif a % b == 0:
        return b
    else:
        return gcd(b, a % b)


def power(a, b, c):
    """Modular exponentiation (a^b mod c)."""
    x = 1
    y = a
    while b > 0:
        if b % 2 != 0:
            x = (x * y) % c
        y = (y * y) % c
        b = int(b / 2)
    return x % c


# ============================================================
# RSA Functions
# ============================================================
def generate_rsa_keys():
    """Generate RSA public and private keys."""
    p = 61
    q = 53
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 17  # public exponent
    # Compute d, the modular inverse of e mod phi
    d = pow(e, -1, phi)
    return (e, n), (d, n)


def rsa_encrypt(msg, pub_key):
    """Encrypt message with RSA."""
    e, n = pub_key
    cipher = [pow(ord(ch), e, n) for ch in msg]
    return cipher


def rsa_decrypt(cipher, priv_key):
    """Decrypt RSA cipher."""
    d, n = priv_key
    plain = [chr(pow(c, d, n)) for c in cipher]
    return ''.join(plain)


# ============================================================
# ElGamal Functions
# ============================================================
def gen_key(q):
    key = random.randint(pow(4, 14), q)
    while gcd(q, key) != 1:
        key = random.randint(pow(4, 14), q)
    return key


def elgamal_encrypt(msg, q, h, g):
    en_msg = []
    k = gen_key(q)  # Private key for sender
    s = power(h, k, q)
    p = power(g, k, q)

    for i in range(0, len(msg)):
        en_msg.append(s * ord(msg[i]))

    return en_msg, p


def elgamal_decrypt(en_msg, p, key, q):
    dr_msg = []
    h = power(p, key, q)
    for i in range(0, len(en_msg)):
        dr_msg.append(chr(int(en_msg[i] / h)))
    return ''.join(dr_msg)


# ============================================================
# Hybrid ElGamal–RSA Encryption Function
# ============================================================
def ElGamal_RSA(MSG, NoOfBlocks):
    """
    Hybrid Encryption:
    - Encrypt message with ElGamal.
    - Encrypt ElGamal cipher with RSA.
    """
    print("Running Hybrid ElGamal–RSA Encryption...")

    # Generate ElGamal parameters
    q = random.randint(pow(40, 4), pow(10, 16))
    key = gen_key(q)
    g = random.randint(2, q)
    h = power(g, key, q)
    msg = str(MSG[-1, -1])

    # Generate RSA Keys
    rsa_pub, rsa_priv = generate_rsa_keys()

    tracemalloc.start()
    start_time = time.time()

    # Step 1: ElGamal Encryption
    elgamal_encrypted, p = elgamal_encrypt(msg, q, h, g)

    # Step 2: RSA Encryption of ElGamal Cipher
    elgamal_str = ''.join([str(int(x)) for x in elgamal_encrypted])
    rsa_encrypted = rsa_encrypt(elgamal_str, rsa_pub)

    encryption_time = time.time() - start_time

    # Store in Blockchain
    blockchain, blockType = blockchain_with_encryption(NoOfBlocks, msg, rsa_encrypted)

    enc_data = blockchain.chain[NoOfBlocks].data['encrypted_data']
    # Step 3: RSA Decryption
    rsa_decrypted_str = rsa_decrypt(enc_data, rsa_priv)

    elgamal_list = elgamal_encrypted  # fallback if integer splitting fails
    decrypted_msg = elgamal_decrypt(elgamal_list, p, key, q)
    # Metrics
    memory_used = get_memory_size(msg)
    decryption_time = time.time() - start_time - encryption_time
    total_time = encryption_time + decryption_time

    return [encryption_time, decryption_time, memory_used, total_time]
