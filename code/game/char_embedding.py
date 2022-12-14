import torch
import numpy as np
import torch.nn.functional as F


class BoCharEmbeddingLayer(object):
    def __init__(self, embedding_size, use_cuda=False, pad_val=0.):
        """
        Bag-of-characters embedding layer. Given a batch of lists of lists of token characters, this layer will
            return character embeddings where each slot represents the frequency of that character in the token.
        :param embedding_size: Size of embeddings to return. This should be equal to the char vocab size.
        :param use_cuda: Whether or not to use cuda tensors.
        :param pad_val: Value to use as padding.
        """
        self.embed_size = embedding_size
        self.use_cuda = use_cuda
        self.pad_val = pad_val

    def __call__(self, batch_inputs, max_num_tokens):
        """
        Obtain bag-of-character embeddings for the given batch of inputs.
        :param batch_inputs: List of lists of lists of character IDs.
        :param max_num_tokens: Maximum number of tokens.
        :return: A torch tensor of shape B x T x D (B = batch size, T = max number of tokens, D = embedding size).
        """
        
        char_vecs = [torch.from_numpy(np.asarray([np.bincount(t, minlength=self.embed_size) for t in ex] \
                                                 if len(ex)!=0 else [[0]*self.embed_size])).float()
                     for ex in batch_inputs]

        char_reps = torch.stack([torch.nn.functional.pad(v, (0, 0, 0, max_num_tokens - v.size(0)), value=self.pad_val)
                                 for v in char_vecs], dim=0)
        return char_reps.cuda() if self.use_cuda else char_reps


class CharCNN(torch.nn.Module):
    def __init__(self, embedding_num, embedding_dim, filters):
        """
        Character convolutional neural network layer used to obtain character embeddings.
        :param embedding_num: Number of characters (equivalent to character vocabulary size).
        :param embedding_dim: Size of character embeddings.
        :param filters: List of tuples that specify the convolutional filters to use.
        """
        super(CharCNN, self).__init__()
        self.output_size = sum([x[1] for x in filters])
        self.embedding = torch.nn.Embedding(embedding_num,
                                            embedding_dim,
                                            padding_idx=0,
                                            sparse=True)
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(1, x[1], (x[0], embedding_dim))
                                          for x in filters])

    def forward(self, inputs):
        """
        Forward function of CNN.
        :param inputs: A tensor of size B*T x K (B = batch size, T = max number of tokens,
                        K = max number of characters in any token).
        :return: Character embeddings for each token of the shape B*T x D (B and T are the same as in inputs and
                    D = self.output_size).
        """
        inputs_embed = self.embedding.forward(inputs)
        inputs_embed = inputs_embed.unsqueeze(1)
        conv_outputs = [F.relu(conv.forward(inputs_embed)).squeeze(3)
                        for conv in self.convs]
        max_pool_outputs = [F.max_pool1d(i, i.size(2)).squeeze(2)
                            for i in conv_outputs]
        outputs = torch.cat(max_pool_outputs, 1)
        return outputs


# if __name__ == "__main__":
#     tokens = [["hello", "world", "computer", "science"], ["hello", "lkj", "oiu*A"]]
#     batch_size = len(tokens)
#     char_proc = CharProcessor(max_num_chars=30, ignore_case=True)
#     batch = char_proc.numberize(tokens)
#     max_num_tokens = max([len(entry) for entry in tokens])
#
#     # embedding_layer = BoCharEmbeddingLayer(embedding_size=char_proc.vocab_size())
#     # print(embedding_layer(batch, max_num_tokens))
#
#     batch, _ = char_proc.add_padding(batch, max_num_tokens)
#     filters_str = "2,25;3,25;4,25"
#     cnn_filters = [[int(f.split(',')[0]), int(f.split(',')[1])] for f in filters_str.split(';')]
#     embedding_layer = CharCNN(char_proc.vocab_size(), 10, cnn_filters)
#     out_reps = embedding_layer(torch.LongTensor(batch))
#     print(out_reps.view(batch_size, max_num_tokens, -1))
