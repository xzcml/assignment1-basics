#分词三种方法：字符级：Unicode表太大，vob利用率低 字词级：会出现OOC，字典大   字节级：压缩比低，序列长

#Probelm(unicode1):
# What Unicode character does chr(0) return? 空
# How does this character’s string representation (__repr__()) differ from its printed representation?  chr(0).__repr__() \x00
#print("this is a test" + chr(0) + "string") this is a teststring

#Probelm(unicode2):
#What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? UTF-8编码长度最短,UTF16太多\x00
#utf-8是可变长度编码，常用字符占1个字节，汉字占3个字节。对于汉字就无法解码成功
# def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
#     return "".join([bytes([b]).decode("utf-8") for b in bytestring])
text = "hello"
token = ("\x00","\x32")

print("\n".encode("utf-8"))