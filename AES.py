import numpy as np
from crypto.Cipher import AES as aes
from crypto.Util.Padding import pad, unpad
from crypto.Random import get_random_bytes
import tracemalloc
import time
import sys
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


def encrypt_aes(plaintext, key):
    cipher = aes.new(key, aes.MODE_CBC)
    ciphertext = cipher.encrypt(pad(plaintext.encode(), aes.block_size))
    return ciphertext, cipher.iv

def decrypt_aes(ciphertext, key, iv):
    cipher = aes.new(key, aes.MODE_CBC, iv)
    decrypted_plaintext = unpad(cipher.decrypt(ciphertext), aes.block_size)
    return decrypted_plaintext.decode()


def get_memory_size(obj):
    return sys.getsizeof(obj)

def AES(message, NoOfBlocks):
    if len(message) > 1:
        ENC_time = []
        DEC_Time = []
        mem_size = []
        Compt_Time = []
        for n in range(len(message)):
            msg = str(message[n])
            key = get_random_bytes(32)  # 256-bit key
            tracemalloc.start()
            ct = time.time()
            Encrypted_Data, iv = encrypt_aes(msg, key)
            blockChain, blockType = blockchain_with_encryption(NoOfBlocks, msg, Encrypted_Data)
            ENC_time.append(time.time() - ct)
            mem_size.append(get_memory_size(msg))
            enc_data = blockChain.chain[NoOfBlocks].data['encrypted_data']
            decrypted_data = decrypt_aes(enc_data, key, iv)
            DEC_Time.append(time.time() - ENC_time[n])
            Compt_Time.append(ENC_time[n] + DEC_Time[n])
        ENC_time = np.mean(ENC_time, axis=0)
        DEC_Time = np.mean(DEC_Time, axis=0)
        mem_size = np.mean(mem_size, axis=0)
        Compt_Time = np.mean(Compt_Time, axis=0)
    else:
        plaintext = str(message)
        key = get_random_bytes(32)  # 256-bit key
        tracemalloc.start()
        ct = time.time()
        Encrypted_Data, iv = encrypt_aes(plaintext, key)
        blockChain, blockType = blockchain_with_encryption(NoOfBlocks, plaintext, Encrypted_Data)
        ENC_time = time.time() - ct
        enc_data = blockChain.chain[NoOfBlocks].data['encrypted_data']
        decrypted_data = decrypt_aes(enc_data, key, iv)
        mem_size = get_memory_size(plaintext)
        DEC_Time = time.time() - ENC_time
        Compt_Time = ENC_time + DEC_Time
    return [ENC_time, DEC_Time, mem_size, Compt_Time]
