<div align='center'><font size='20'> Model </font></div>

# On top of Transformer and MegatronModule
* Architected with `language_model`(Transformer), embedding, linear, activation, layernorm and etc
* Post processing for the last pipeline layer
* Forward
* Customized `state_dict` and `load_state_dict`
## GPTModel
`megatron/model/gpt_model.py`

<details> 
    <summary>Code for GPTModel</summary>  

```Python
class GPTModel(MegatronModule):
    """GPT-2 Language model."""

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        super(GPTModel, self).__init__()
        args = get_args()

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            init_method=init_method_normal(args.init_method_std),
            scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers),
            pre_process=self.pre_process,
            post_process=self.post_process)

        self.initialize_word_embeddings(init_method_normal)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask, labels=None,
                tokentype_ids=None, inference_params=None):

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            inference_params=inference_params)

        if self.post_process:
            return post_language_model_processing(
                lm_output, labels,
                self.word_embeddings_weight(),
                self.parallel_output,
                self.fp16_lm_cross_entropy)
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)
```
</details>

## BertModel
## T5Model

# MegatronModule
`megatron/model/module.py`
* Embedding with pipeline aware
* Interfaces for conversion between fp32 and float16(fp16, bf16 etc)

## initialize_word_embeddings
* Parameters are shared between the word embeddings layers, and the heads at the end of the model
* For `mpu.is_pipeline_last_stage() and not self.pre_process`
  - `self.word_embeddings` = [`mpu.VocabParallelEmbedding`](mpu.md#vocabparallelembedding) and initialize weight to 0.0
* For `not mpu.is_pipeline_first_stage(ignore_virtual=True) and self.pre_process`
  - `self.language_model.embedding.zero_parameters()`
* For [`mpu.is_rank_in_embedding_group()`](mpu.md#embeddinggroup--embeddingglobalranks)
  - `all_reduce` [`self.word_embeddings_weight().data`](#word_embeddings_weight)
* For `mpu.is_rank_in_position_embedding_group() and args.pipeline_model_parallel_split_rank is not None`
  - `all_reduce` `self.language_model.embedding.position_embeddings.data`
  
  <details> 
    <summary>Code for initialize_word_embeddings</summary>  

  ```Python
  def initialize_word_embeddings(self, init_method_normal):
      args = get_args()
      if not self.share_word_embeddings:
          raise Exception('initialize_word_embeddings() was called but '
                          'share_word_embeddings is false')

      # This function just initializes the word embeddings in the final stage
      # when we are using pipeline parallelism. Nothing to do if we aren't
      # using pipeline parallelism.
      if args.pipeline_model_parallel_size == 1:
          return

      # Parameters are shared between the word embeddings layers, and the
      # heads at the end of the model. In a pipelined setup with more than
      # one stage, the initial embedding layer and the head are on different
      # workers, so we do the following:
      # 1. Create a second copy of word_embeddings on the last stage, with
      #    initial parameters of 0.0.
      # 2. Do an all-reduce between the first and last stage to ensure that
      #    the two copies of word_embeddings start off with the same
      #    parameter values.
      # 3. In the training loop, before an all-reduce between the grads of
      #    the two word_embeddings layers to ensure that every applied weight
      #    update is the same on both stages.
      if mpu.is_pipeline_last_stage() and \
              not self.pre_process:
          assert not mpu.is_pipeline_first_stage()
          self._word_embeddings_for_head_key = 'word_embeddings_for_head'
          # set word_embeddings weights to 0 here, then copy first
          # stage's weights using all_reduce below.
          self.word_embeddings = mpu.VocabParallelEmbedding(
              args.padded_vocab_size, args.hidden_size,
              init_method=init_method_normal(args.init_method_std))
          self.word_embeddings.weight.data.fill_(0)
          self.word_embeddings.weight.shared = True

      # Zero out initial weights for decoder embedding.
      # NOTE: We don't currently support T5 with the interleaved schedule.
      if not mpu.is_pipeline_first_stage(ignore_virtual=True) and \
              self.pre_process:
          self.language_model.embedding.zero_parameters()

      # Ensure that first and last stages have the same initial parameter
      # values.
      if torch.distributed.is_initialized():
          if mpu.is_rank_in_embedding_group():
              torch.distributed.all_reduce(self.word_embeddings_weight().data,
                                           group=mpu.get_embedding_group())

          # Ensure that encoder(first stage) and decoder(split stage) position 
          # embeddings have the same initial parameter values
          # NOTE: We don't currently support T5 with the interleaved schedule.
          if mpu.is_rank_in_position_embedding_group() and \
                  args.pipeline_model_parallel_split_rank is not None:
              # TODO: Support tokentype embedding.
              self.language_model.embedding.cuda()
              position_embeddings = self.language_model.embedding.position_embeddings
              torch.distributed.all_reduce(position_embeddings.weight.data,
                                           group=mpu.get_position_embedding_group())
      else:
          print("WARNING! Distributed processes aren't initialized, so "
                "word embeddings in the last layer are not initialized. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong.")
  ```
  </details>

## word_embeddings_weight
* If `self.pre_process`: return `self.language_model.embedding.word_embeddings.weight`
* Else: return `self.word_embeddings.weight`

  <details> 
    <summary>Code for word_embeddings_weight</summary>  

  ```Python
  def word_embeddings_weight(self):
      if self.pre_process:
          return self.language_model.embedding.word_embeddings.weight
      else:
          if not self.share_word_embeddings:
              raise Exception('word_embeddings_weight() called for last '
                              'stage, but share_word_embeddings is false')
          return self.word_embeddings.weight
  ```
  </details>


<details> 
    <summary>Code for MegatronModule</summary>  

```Python
class MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    def __init__(self, share_word_embeddings=True):
        super(MegatronModule, self).__init__()
        self.share_word_embeddings = share_word_embeddings


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(destination, prefix, keep_vars)


    def word_embeddings_weight(self):
        if self.pre_process:
            return self.language_model.embedding.word_embeddings.weight
        else:
            if not self.share_word_embeddings:
                raise Exception('word_embeddings_weight() called for last '
                                'stage, but share_word_embeddings is false')
            return self.word_embeddings.weight


    def initialize_word_embeddings(self, init_method_normal):
        args = get_args()
        if not self.share_word_embeddings:
            raise Exception('initialize_word_embeddings() was called but '
                            'share_word_embeddings is false')

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism. Nothing to do if we aren't
        # using pipeline parallelism.
        if args.pipeline_model_parallel_size == 1:
            return

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        if mpu.is_pipeline_last_stage() and \
                not self.pre_process:
            assert not mpu.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = 'word_embeddings_for_head'
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.word_embeddings = mpu.VocabParallelEmbedding(
                args.padded_vocab_size, args.hidden_size,
                init_method=init_method_normal(args.init_method_std))
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True

        # Zero out initial weights for decoder embedding.
        # NOTE: We don't currently support T5 with the interleaved schedule.
        if not mpu.is_pipeline_first_stage(ignore_virtual=True) and \
                self.pre_process:
            self.language_model.embedding.zero_parameters()

        # Ensure that first and last stages have the same initial parameter
        # values.
        if torch.distributed.is_initialized():
            if mpu.is_rank_in_embedding_group():
                torch.distributed.all_reduce(self.word_embeddings_weight().data,
                                             group=mpu.get_embedding_group())

            # Ensure that encoder(first stage) and decoder(split stage) position 
            # embeddings have the same initial parameter values
            # NOTE: We don't currently support T5 with the interleaved schedule.
            if mpu.is_rank_in_position_embedding_group() and \
                    args.pipeline_model_parallel_split_rank is not None:
                # TODO: Support tokentype embedding.
                self.language_model.embedding.cuda()
                position_embeddings = self.language_model.embedding.position_embeddings
                torch.distributed.all_reduce(position_embeddings.weight.data,
                                             group=mpu.get_position_embedding_group())
        else:
            print("WARNING! Distributed processes aren't initialized, so "
                  "word embeddings in the last layer are not initialized. "
                  "If you are just manipulating a model this is fine, but "
                  "this needs to be handled manually. If you are training "
                  "something is definitely wrong.")


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_float16(val, float16_convertor):
    """Convert fp32 `val` to fp16/bf16"""
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES):
            val = float16_convertor(val)
        return val
    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val
    return conversion_helper(val, float_conversion)
```
</details>

# TransformerLanguageModel
`megatron/model/language_model.py`

## initialize
* Initialize Embedding if `self.pre_process`
* Initialize encoder or decoder to ParallelTransformer according to `self.add_encoder` and `self.add_decoder`
* Initialize Pooler if `self.post_process and self.add_pooler`
  <details> 
    <summary>Code for __init__</summary>  

  ```Python
  def __init__(self,
               init_method,
               output_layer_init_method,
               encoder_attn_mask_type,
               num_tokentypes=0,
               add_encoder=True,
               add_decoder=False,
               decoder_attn_mask_type=AttnMaskType.causal,
               add_pooler=False,
               pre_process=True,
               post_process=True):
      super(TransformerLanguageModel, self).__init__()
      args = get_args()

      self.pre_process = pre_process
      self.post_process = post_process
      self.hidden_size = args.hidden_size
      self.num_tokentypes = num_tokentypes
      self.init_method = init_method
      self.add_encoder = add_encoder
      self.encoder_attn_mask_type = encoder_attn_mask_type
      self.add_decoder = add_decoder
      self.decoder_attn_mask_type = decoder_attn_mask_type
      self.add_pooler = add_pooler
      self.encoder_hidden_state = None

      # Embeddings.
      if self.pre_process:
          self.embedding = Embedding(self.hidden_size,
                                     args.padded_vocab_size,
                                     args.max_position_embeddings,
                                     args.hidden_dropout,
                                     self.init_method,
                                     self.num_tokentypes)
          self._embedding_key = 'embedding'

      # Transformer.
      # Encoder (usually set to True, False if part of an encoder-decoder
      # architecture and in encoder-only stage).
      if self.add_encoder:
          self.encoder = ParallelTransformer(
              self.init_method,
              output_layer_init_method,
              self_attn_mask_type=self.encoder_attn_mask_type,
              pre_process=self.pre_process,
              post_process=self.post_process
          )
          self._encoder_key = 'encoder'
      else:
          self.encoder = None

      # Decoder (usually set to False, True if part of an encoder-decoder
      # architecture and in decoder-only stage).
      if self.add_decoder:
          self.decoder = ParallelTransformer(
              self.init_method,
              output_layer_init_method,
              layer_type=LayerType.decoder,
              self_attn_mask_type=self.decoder_attn_mask_type,
              pre_process=self.pre_process,
              post_process=self.post_process)
          self._decoder_key = 'decoder'
      else:
          self.decoder = None

      if self.post_process:
          # Pooler.
          if self.add_pooler:
              self.pooler = Pooler(self.hidden_size, self.init_method)
              self._pooler_key = 'pooler'
  ```
  </details>

## forward

  <details> 
    <summary>Code for forward</summary>  

  ```Python
  def forward(self, enc_input_ids, enc_position_ids, enc_attn_mask,
              dec_input_ids=None, dec_position_ids=None, dec_attn_mask=None,
              enc_dec_attn_mask=None, tokentype_ids=None,
              inference_params=None,
              pooling_sequence_index=0,
              enc_hidden_states=None, output_enc_hidden=False):

      # Encoder embedding.
      if self.pre_process:
          encoder_input = self.embedding(enc_input_ids, enc_position_ids,
                                         tokentype_ids=tokentype_ids)
      else:
          encoder_input = None

      # Run encoder.
      if enc_hidden_states is None:
          if self.encoder is not None:
              encoder_output = self.encoder(
                  encoder_input,
                  enc_attn_mask,
                  inference_params=inference_params)
          else:
              encoder_output = self.encoder_hidden_state
      else:
          encoder_output = enc_hidden_states.to(encoder_input.dtype)

      if self.post_process:
          if self.add_pooler:
              pooled_output = self.pooler(encoder_output,
                                          pooling_sequence_index)

      # output_enc_hidden refers to when we just need the encoder's
      # output. For example, it is helpful to compute
      # similarity between two sequences by average pooling
      if not self.add_decoder or output_enc_hidden:
          if self.add_pooler and self.post_process:
              return encoder_output, pooled_output
          else:
              return encoder_output

      # Decoder embedding.
      if self.pre_process:
          decoder_input = self.embedding(dec_input_ids,
                                         dec_position_ids)
      else:
          decoder_input = None

      # Run decoder.
      decoder_output = self.decoder(
          decoder_input,
          dec_attn_mask,
          encoder_output=encoder_output,
          enc_dec_attn_mask=enc_dec_attn_mask,
          inference_params=inference_params)

      if self.add_pooler and self.post_process:
          return decoder_output, encoder_output, pooled_output
      else:
          return decoder_output, encoder_output
  ```
  </details>