import cProfile
from collections import Counter, defaultdict
import pickle
import time
import regex as re
import os
from typing import BinaryIO
from multiprocessing import Manager, Process, Queue

#PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\p{L}\p{N}\s]+"""
# PAT = r"""'(?:[sdmt]|ll|ve|re)
#         | ?\p{L}+
#         | ?\p{N}+
#         | ?[^\s\p{L}\p{N}]+
#         |\n+
#         |[ \t\f\v]+"""
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
text = f"""low<|endoftext|><|endoftext|>low<|endoftext|>low low low lower lower widest widest widest newest newest newest newest newest newest"""
special_tokens = ["<|endoftext|>","<|pad|>"]
vocab_size = 512


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

class BPEtrainer:
    def __init__(self,vocab_size,special_tokens,fre_table):
        self.vocab = self.init_vocab(special_tokens)
        self.num_merge = vocab_size - 256 - len(special_tokens)
        self.special_tokens = sorted(special_tokens,key = lambda x: len(x),reverse=True)
        self.fre_table = fre_table
        self.split_word = {word: self.word_to_bytes(word) for word in fre_table.keys()}
        self.pairs,self.pairs_to_word = self.pair_count(fre_table)
        self.merges = []

    def word_to_bytes(self,word_encode):
        return tuple(word_encode[i:i+1] for i in range(len(word_encode)))
    
    def init_vocab(self,special_tokens:list[str]) -> dict[int, bytes]:
        vocab = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for i,token in enumerate(special_tokens,start=256):
            vocab[i] = token.encode("utf-8")
        return vocab
    def pair_count(self,fre_table:dict[bytes, int]) :
        pairs = Counter()
        pairs_to_word = defaultdict(set)
        for token,fre in fre_table.items():
            for i in range(len(token)-1):
                pair = (token[i:i+1],token[i+1:i+2])
                pairs[pair] = pairs.get(pair,0) + fre
                pairs_to_word[pair].add(token)            
        return pairs,pairs_to_word
    
    def update_pairs(self,new_pair,old_pair,word,fre):
        self.pairs_to_word[new_pair].add(word)
        #self.pairs_to_word[old_pair].remove(word)
        self.pairs[new_pair] = self.pairs.get(new_pair,0) + fre
        if old_pair in self.pairs:
            self.pairs[old_pair] -= fre
            if self.pairs[old_pair] <= 0:
                del self.pairs[old_pair]

    def merge_pair(self,merge_tuple) :
        new_words = list(self.pairs_to_word[merge_tuple])
        del self.pairs[merge_tuple]
        for word in new_words:
            n = len(self.split_word[word])
            token = self.split_word[word]
            new_split = []
            i=0
            fre = self.fre_table[word]
            while i < n:
                if i < n-1 and (token[i],token[i+1]) == merge_tuple:
                    new_split.append(token[i]+token[i+1])  
                    if i>0:
                        old_pair = (token[i-1],token[i])
                        new_pair = (token[i-1],token[i]+token[i+1])
                        self.update_pairs(new_pair,old_pair,word,fre)
                    if i<n-2:
                        old_pair = (token[i+1],token[i+2])
                        new_pair = (token[i]+token[i+1],token[i+2])
                        self.update_pairs(new_pair,old_pair,word,fre)
                    i+=2
                else:
                    new_split.append(token[i])
                    i+=1
            self.split_word[word] = new_split

    def train_bpe(self):

        for i in range(self.num_merge):
            if len(self.pairs_to_word.keys()) <=1:
                break
            max_freq = max(self.pairs.values())
            merge_list = [k for k,v in self.pairs.items() if v == max_freq]
            merge_tuple = max([k for k,v in self.pairs.items() if v == max_freq],key = lambda x: x)
        
            self.merges.append(merge_tuple)
            tmp_n = len(self.vocab)
            self.vocab[tmp_n] = merge_tuple[0] + merge_tuple[1]

            self.merge_pair(merge_tuple)
        
        return self.vocab,self.merges

        
def split_by_special_tokens(text:str,special_tokens:list[str]) -> list[str]:
    pattern = "|".join(re.escape(token) for token in special_tokens) if special_tokens else r'(?!)'
    return re.split(f'({pattern})',text)

def pre_tokenize(input_path:str,special_tokens:list[str],start:int,end:int,queue:Queue) -> dict[tuple[bytes], int]:
    with open(input_path,"rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    split_text = split_by_special_tokens(text,special_tokens)
    fre_table = Counter()
    for chunk in split_text:
        if chunk not in special_tokens:
            chunk = re.finditer(PAT,chunk)
            for match in chunk:
                token = match.group(0)
                token_encode = token.encode("utf-8")
                fre_table[token_encode] += 1
    queue.put(fre_table)



def train_merge(input_path,special_tokens,vocab_size) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    fre_table = Counter()
    manager = Manager()
    queue = manager.Queue()
    processes = []
    with open(input_path,"rb") as f:
        num_processes = os.cpu_count() or 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        p = Process(target=pre_tokenize,args=(input_path,special_tokens,start,end,queue))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    for _ in range(len(processes)):
        fre_table += queue.get()
    BPE_train = BPEtrainer(vocab_size,special_tokens,fre_table)
    return BPE_train.train_bpe()
    
       

if __name__ == "__main__":
    input_path = "tests/fixtures/tinystories_sample_5M.txt"
    # start_time = time.time()
    # _, _ = train_merge(
    #     input_path=input_path,
    #     vocab_size=500,
    #     special_tokens=["<|endoftext|>"],
    # )
    # end_time = time.time()
    vocab,merges = train_merge(input_path=input_path,vocab_size=1000,special_tokens=["<|endoftext|>"])
    with open("tests/_snapshots/test_train_bpe_special_tokens.pkl", "rb") as f:
        expected_data = pickle.load(f)
    for i in range(len(merges)):
        if merges[i] != expected_data["merges"][i]:
            print(i,merges[i],expected_data["merges"][i])
    print()
    print(expected_data["vocab_values"])
    print(set(vocab.values())==expected_data["vocab_values"])
    
    
    