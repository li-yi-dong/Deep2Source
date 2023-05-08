<div align='center'><font size='20'> language model </font></div>

# Embedding
## initialization
- word embeddings = `tensor_parallel.VocabParallelEmbedding`
- position embeddings = `torch.nn.Embedding(max_sequence_length, self.hidden_size)`
- tokentype_embeddings = `torch.nn.Embedding(self.num_tokentypes, self.hidden_size)`
<details> 
    <summary>Code for Embedding.__init__</summary>  

```Python
def __init__(self,
                hidden_size,
                vocab_size,
                max_sequence_length,
                embedding_dropout_prob,
                init_method,
                num_tokentypes=0):
    super(Embedding, self).__init__()

    self.hidden_size = hidden_size
    self.init_method = init_method
    self.num_tokentypes = num_tokentypes

    args = get_args()

    # Word embeddings (parallel).
    self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
        vocab_size, self.hidden_size,
        init_method=self.init_method,
        params_dtype=args.params_dtype,
        use_cpu_initialization=args.use_cpu_initialization,
        perform_initialization=args.perform_initialization
    )
    self._word_embeddings_key = 'word_embeddings'

    # Position embedding (serial).
    self.add_position_embedding = args.add_position_embedding
    if self.add_position_embedding:
        self.position_embeddings = torch.nn.Embedding(
            max_sequence_length, self.hidden_size)
        self._position_embeddings_key = 'position_embeddings'
        # Initialize the position embeddings.
        if args.perform_initialization:
            self.init_method(self.position_embeddings.weight)

    # Token type embedding.
    # Add this as an optional field that can be added through
    # method call so we can load a pretrain model without
    # token types and add them as needed.
    self._tokentype_embeddings_key = 'tokentype_embeddings'
    if self.num_tokentypes > 0:
        self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes,
                                                        self.hidden_size)
        # Initialize the token-type embeddings.
        if args.perform_initialization:
            self.init_method(self.tokentype_embeddings.weight)
    else:
        self.tokentype_embeddings = None

    self.fp32_residual_connection = args.fp32_residual_connection 
    self.sequence_parallel = args.sequence_parallel
    # Embeddings dropout
    self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
```
</details>

## forward
