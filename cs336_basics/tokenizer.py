# import regex as re

# class Tokenizer:
#     PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
#     def __init__(self, vocab, merges, special_tokens=None):
#         """
#         Construct a tokenizer from a given
#         vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept
#         the following parameters:
#         vocab: dict[int, bytes]
#         merges: list[tuple[bytes, bytes]]
#         special_tokens: list[str] | None = None
#         """
#         self.vocab = vocab if special_tokens is not None else self.init_vocab(vocab,special_tokens)
#         self.merges = merges
#         self.special_tokens = special_tokens if special_tokens is not None else []
#         self.vocab_inv = {v: k for k, v in self.vocab.items()}

#         self.cache_word = {}

#     def init_vocab(self,vocab:dict[int, bytes],special_tokens:list[str]) -> dict[int, bytes]:
#         start_index = len(vocab)
#         for token in special_tokens:
#             if token.encode("utf-8") not in vocab.values():
#                 vocab[start_index] = token.encode("utf-8")
#                 start_index += 1
#         return vocab
    
#     @classmethod
#     def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
#         """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
#             (in the same format that your BPE training code output) and (optionally) a list of special tokens. 
#         This method should accept the following additional parameters:
#             vocab_filepath: str
#             merges_filepath: str
#             special_tokens: list[str] | None = None
#         """
#         vocab = {}
#         with open(vocab_filepath, 'rb') as f:
#             # 读取词汇表大小
#             vocab_size_bytes = f.read(4)
#             vocab_size = int.from_bytes(vocab_size_bytes, byteorder='little')
            
#             # 读取每个token: <id(4字节)><长度(4字节)><token内容(bytes)>
#             for _ in range(vocab_size):
#                 token_id_bytes = f.read(4)
#                 token_id = int.from_bytes(token_id_bytes, byteorder='little')
                
#                 token_len_bytes = f.read(4)
#                 token_len = int.from_bytes(token_len_bytes, byteorder='little')
                
#                 token = f.read(token_len)
#                 vocab[token_id] = token
        
#         # 读取merges
#         merges = []
#         with open(merges_filepath, 'rb') as f:
#             # 读取合并规则数量
#             merges_count_bytes = f.read(4)
#             merges_count = int.from_bytes(merges_count_bytes, byteorder='little')
            
#             # 读取每个合并规则: <第一部分长度(4字节)><第一部分内容(bytes)><第二部分长度(4字节)><第二部分内容(bytes)>
#             for _ in range(merges_count):
#                 first_len_bytes = f.read(4)
#                 first_len = int.from_bytes(first_len_bytes, byteorder='little')
                
#                 first = f.read(first_len)
                
#                 second_len_bytes = f.read(4)
#                 second_len = int.from_bytes(second_len_bytes, byteorder='little')
                
#                 second = f.read(second_len)
                
#                 merges.append((first, second))
        
#         return cls(vocab, merges, special_tokens)

#     def pretokenize(self,text: str) -> list[bytes] :
#         pattern = "|".join(re.escape(token) for token in self.special_tokens) if self.special_tokens else r'(?!)'
#         split_text = re.split(f'({pattern})',text)
#         result = []
#         for token in split_text:
#             if token in self.special_tokens:
#                 result.append(token.encode("utf-8"))
#             else:
#                 chunks = re.finditer(self.PAT,token)
#                 for match in chunks:
#                     result.append(match.group(0).encode("utf-8"))

#         return result

#     def calucate_word_token(self,word:bytes) -> list[int]:
#         result = []
#         word_byte = [word[i:i+1] for i in range(len(word))]
#         min_merge_index = len(self.vocab)
        
#         while len(word_byte) > 1:
#             min_i = -1
#             min_pair = None
#             for i,pairs in enumerate(zip(word_byte[:-1],word_byte[1:])):
#                 if pairs in self.merges:
#                     if self.vocab_inv[pairs[0]+pairs[1]] < min_merge_index:
#                         min_merge_index = self.vocab_inv[pairs[0]+pairs[1]]
#                         min_pair = pairs
#                         min_i = i
#             if min_i == -1:
#                 break
#             word_byte = word_byte[:min_i] + [min_pair[0]+min_pair[1]] + word_byte[min_i+2:]
#         for token in word_byte:
#             if token in self.vocab_inv:
#                 result.append(self.vocab_inv[token])
#             else:
#                 result.append
#         return result
                

#     def encode(self, text: str) -> list[int] :
#         """ Encode an input text into a sequence of token IDs."""
#         tokens = []
#         text_encodes = self.pretokenize(text)
#         for text_encode in text_encodes:
#             if text_encode in self.vocab_inv:
#                 tokens.append(self.vocab_inv[text_encode])
#             elif text_encode in self.cache_word:
#                 tokens.extend(self.cache_word[text_encode])
#             else:
#                 ids = self.calucate_word_token(text_encode)
#                 tokens.extend(ids)
#                 self.cache_word[text_encode] = ids
                
#         return tokens



#     def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int] :
#         """ Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
#             This is required for memory-effcient tokenization of large files that we cannot directly load into memory.
#         """
#         words_iter = self.pretokenize(iterable)
#         for word in words_iter:
#             if word in self.vocab_inv:

#                 yield self.vocab_inv[word]
#             elif word in self.cache_word:
#                 yield from self.cache_word[word]
#             else:
#                 token_ids = self.calucate_word_token(word)
#                 self.cache_word[word] = token_ids
#                 yield from token_ids

#     def decode(self, ids: list[int]) -> str :
#         """Decode a sequence of token IDs into text.
#             To test your Tokenizer against our provided tests, you will first need to implement the test adapter
#             at [adapters.get_tokenizer]. Then, run uv run pytest tests/test_tokenizer.py. 
#             Your implementation should be able to pass all tests.
#         """
#         text_bytes = b""
#         for id in ids:
#             if id in self.vocab:
#                 text_bytes += self.vocab[id]
#             else:
#                 raise ValueError(f"Token ID {id} not found in vocabulary.")
#         return text_bytes.decode('utf-8', errors='ignore')

    