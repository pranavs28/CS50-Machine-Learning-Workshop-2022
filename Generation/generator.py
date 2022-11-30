#tutorial from https://colab.research.google.com/drive/1VLG8e7YSEwypxU-noRNhsv5dW4NfTGce?usp=sharing

import gpt_2_simple as gpt2
from datetime import datetime

gpt2.download_gpt2(model_name="124M")

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset="shakespeare.txt",
              model_name='124M',
              steps=1000,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=200,
              save_every=500
              )


sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name='run1')

gpt2.generate(sess, run_name='run1')