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
            "encrypted_ElGamal": encrypted_ElGamal,
        }
        new_block = Block(i + 1, block_data, blockchain.chain[-1].hash)
        blockchain.add_block(new_block)
        blockType.append(1 if i % 5 == 0 else 0)
    return blockchain, blockType


def get_memory_size(obj):
    return sys.getsizeof(obj)


def gcd(a, b):
    if a < b:
        return gcd(b, a)
    elif a % b == 0:
        return b
    else:
        return gcd(b, a % b)


def gen_key(q):
    if type(q) != int:
        # Convert numpy array to binary string
        binary_string = ''.join(map(str, q))
        # Convert binary string to integer
        q = int(binary_string)
    key = random.randint(pow(4, 14), q)
    while gcd(q, key) != 1:
        key = random.randint(pow(4, 14), q)

    return key


# Modular exponentiation
def power(a, b, c):
    x = 1
    y = a

    while b > 0:
        if b % 2 != 0:
            x = (x * y) % c
        y = (y * y) % c
        b = int(b / 2)

    return x % c


# Asymmetric encryption
def encrypt(msg, q, h, g):
    en_msg = []

    k = gen_key(q)  # Private key for sender
    s = power(h, k, q)
    p = power(g, k, q)

    for i in range(0, len(msg)):
        en_msg.append(msg[i])

    print("g^k used : ", p)
    print("g^ak used : ", s)
    for i in range(0, len(en_msg)):
        en_msg[i] = s * ord(en_msg[i])

    return en_msg, p


def decrypt(en_msg, p, key, q):
    dr_msg = []
    h = power(p, key, q)
    for i in range(0, len(en_msg)):
        dr_msg.append(chr(int(en_msg[i] / h)))

    return dr_msg


def ElGamal_encryption(MSG, NoOfBlocks, key=None, q=None):

    if key is None or q is None:
        q = random.randint(pow(40, 4), pow(10, 16))
        key = gen_key(q)
    if type(q) != int:
        binary_string = ''.join(map(str, q))
        q = int(binary_string)
    msg = str(MSG[-1, -1])

    g = random.randint(2, q)
    h = power(g, key, q)

    tracemalloc.start()
    ct = time.time()

    en_msg, p = encrypt(msg, q, h, g)
    ENC_time = time.time() - ct

    blockChain, blockType = blockchain_with_encryption(NoOfBlocks, msg, en_msg)

    dr_msg = decrypt(blockChain.chain[NoOfBlocks].data['encrypted_ElGamal'], p, key, q)
    dmsg = ''.join(dr_msg)

    mem_size = get_memory_size(msg)
    DEC_Time = time.time() - ENC_time
    Compt_Time = ENC_time + DEC_Time
    return [ENC_time, DEC_Time, mem_size, Compt_Time]

