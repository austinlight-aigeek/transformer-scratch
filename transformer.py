import torch.nn as nn
from encoder import Encoder, EncoderBlock
from decoder import Decoder, DecoderBlock
from input_embeddings import InputEmbeddings
from positional_encoding import PositionalEncoding
from projection_layer import ProjectionLayer
from multihead_attention import MultiHeadAttentionBlock
from feedforward_network import FeedForwardBlock


# Creating the Transformer Architecture
class Transformer(nn.Module):
    # This takes in the encoder and decoder, as well the embeddings for the source and target language.
    # It also takes in the Positional Encoding for the source and target language, as well as the projection layer
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            src_embed: InputEmbeddings,
            tgt_embed: InputEmbeddings,
            src_pos: PositionalEncoding,
            tgt_pos: PositionalEncoding,
            projection_layer: ProjectionLayer
    ) -> None:

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # Encoder
    def encode(self, src, src_mask):
        src = self.src_embed(src)  # Applying source embeddings to the input source language
        src = self.src_pos(src)  # Applying source positional encoding to the source embeddings

        # Returning the source embeddings plus a source mask to prevent attention to certain elements
        return self.encoder(src, src_mask)

    # Decoder
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)  # Applying target embeddings to the input target language (tgt)
        tgt = self.tgt_pos(tgt)  # Applying target positional encoding to the target embeddings

        # Returning the target embeddings, the output of the encoder, and both source and target masks
        # The target mask ensures that the model won't 'see' future elements of the sequence
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    # Applying Projection Layer with the Softmax function to the Decoder output
    def project(self, x):
        return self.projection_layer(x)


# Building & Initializing Transformer

# Define function and its parameter, including model dimension, number of encoder and decoder stacks, heads, etc.
def build_transformer(
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048
) -> Transformer:
    # Creating Embedding layers
    # Source language (Source Vocabulary to 512-dimensional vectors)
    src_embed = InputEmbeddings(d_model, src_vocab_size)

    # Target language (Target Vocabulary to 512-dimensional vectors)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Creating Positional Encoding layers
    # Positional encoding for the source language embeddings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)

    # Positional encoding for the target language embeddings
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Creating EncoderBlocks
    encoder_blocks = []  # Initial list of empty EncoderBlocks

    for _ in range(N):  # Iterating 'N' times to create 'N' EncoderBlocks (N = 6)
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # Self-Attention
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)  # FeedForward

        # Combine layers into an EncoderBlock
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)  # Appending EncoderBlock to the list of EncoderBlocks

    # Creating DecoderBlocks
    decoder_blocks = []  # Initial list of empty DecoderBlocks
    for _ in range(N):  # Iterating 'N' times to create 'N' DecoderBlocks (N = 6)
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # Self-Attention
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # Cross-Attention
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)  # FeedForward

        # Combining layers into a DecoderBlock
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout
        )

        decoder_blocks.append(decoder_block)  # Appending DecoderBlock to the list of DecoderBlocks

    # Creating the Encoder and Decoder by using the EncoderBlocks and DecoderBlocks lists
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Creating projection layer
    # Map the output of Decoder to the Target Vocabulary Space
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Creating the transformer by combining everything above
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer  # Assembled and initialized Transformer. Ready to be trained and validated!
