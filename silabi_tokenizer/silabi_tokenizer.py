from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

class KiswahiliSilabiTokenizer(PreTrainedTokenizerFast):
    def __init__(self, tokenizer,unk_token="[UNK]",sos_token="[SOS]",eos_token="[EOS]",space_token="[SPACE]", **kwargs):
        super().__init__(tokenizer_object=tokenizer, **kwargs)
        self._vocab = tokenizer.get_vocab()
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.space_token = space_token

      # Add special tokens to vocab if they are not already present
        if self.sos_token not in self._vocab:
            self._vocab[self.sos_token] = len(self._vocab)
        if self.eos_token not in self._vocab:
            self._vocab[self.eos_token] = len(self._vocab)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer = Tokenizer.from_file(f"{pretrained_model_name_or_path}/tokenizer.json")
        return cls(tokenizer, **kwargs)

    def _encode_with_byte_fallback(self, text):
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            # Try to match the longest syllable first
            for j in range(len(text), i, -1):
                syllable_candidate = text[i:j]
                if syllable_candidate in self._vocab:
                    tokens.append(syllable_candidate)
                    i = j
                    matched = True
                    break
            # If no syllable matched, fallback to byte encoding
            if not matched:
                if text[i] == " ":
                  tokens.append(self.space_token)
                  i += 1
                else:
                  tokens.extend(self.unk_token)
                  i += 1
        return tokens

    def tokenize(self, text,**kwargs):
        handle_whitespace = kwargs.get("handle_whitespace", True)
        tokens = [self.sos_token]  # Start of sentence token
        for word in text.split(" "):
            tokens.extend(self._encode_with_byte_fallback(word))
            if handle_whitespace:
              tokens.extend(self._encode_with_byte_fallback(" "))
        tokens.append(self.eos_token)  # End of sentence token
        return tokens

    def encode(self, text, **kwargs):
        tokens = self.tokenize(text,**kwargs)
        encoding = super().encode(" ".join(tokens), **kwargs)
        return encoding

    def tokens_to_sentence(self,tokens):
      for token in tokens:
        token = token.replace(" ", "")
      sentence = "".join(tokens)
      sentence = sentence.replace(self.eos_token, "")
      sentence = sentence.replace(self.sos_token, "")
      sentence = sentence.replace(self.space_token," ")
      return sentence