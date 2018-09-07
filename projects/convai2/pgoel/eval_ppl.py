# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import create_agent, create_agents_from_shared
from parlai.core.build_data import download_models
from projects.convai2.build_dict import build_dict
from projects.convai2.eval_ppl import setup_args, eval_ppl
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser
from parlai.core.thread_utils import SharedTable
from parlai.core.utils import Timer, round_sigfigs, no_lock
from parlai.core.worlds import World, create_task
from torch.autograd import Variable
from language_model import MyLanguageModelAgent
import torch.nn.functional as F
import math


class LanguageModelEntry(MyLanguageModelAgent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if shared:
            self.probs = shared['probs']
        else:
            # default minimum probability mass for all tokens
            self.probs = {k: 1e-7 for k in build_dict().keys()}

    def share(self):
        shared = super().share()
        shared['probs'] = self.probs.copy()
        return shared

    def next_word_probability(self, partial_out):
        """Return probability distribution over next words given an partial
        true output. This is used to calculate the per-word perplexity.
        Arguments:
        partial_out -- previous "true" words
        Returns a dict, where each key is a word and each value is a probability
        score for that word. Unset keys assume a probability of zero.
        e.g.
        {'text': 'Run test program.'}, ['hello'] => {'world': 1.0}
        """
        #print("partial_out : ", partial_out)
        obs = self.observation
        if not hasattr(self, 'last_text'):
            self.last_text = None
            self.reset_next = False
        if obs['text'] != self.last_text:
            if self.reset_next:
                # reset hidden state for new episodes
                self.hidden = self.model.init_hidden(1)
                self.reset_next = False
            self.seen = False
            self.last_text = obs.get('text')

        #self.model.eval()

        if obs['episode_done'] == True:
            self.reset_next = True
        else:
            self.reset_next = False


        if len(partial_out) == 0:
            # first observe 'PERSON2' token
            obs['eval_labels'] = ('PERSON2',)
        else:
            # feed in words one at a time
            obs['eval_labels'] = (partial_out[-1],)
       
        #print(obs['text'])
        data_list, targets_list, labels, valid_inds, y_lens = self.vectorize([obs], self.opt['seq_len'], False)
        #print(data_list)
        #print(targets_list)
        data = data_list[0]
        targets = targets_list[0]

        a = []
        if not self.seen:
            for i in range(data.size()[1]):
                a.append(self.dict[data[0][i].item()])
            output, hidden = self.model(data.transpose(0,1), self.hidden)
            self.hidden = self.repackage_hidden(hidden)
            # feed in end tokens
            a.append(self.dict[self.ends[:1].view(1, 1).item()])
            output, hidden = self.model(Variable(self.ends[:1].view(1,1)), self.hidden)
            # feed in person2 tokens
            a.append(self.dict[targets.select(1, 0).item()])
            output, hidden = self.model(targets.select(1,0).view(1, 1), self.hidden)
            self.hidden = self.repackage_hidden(hidden)
            output_flat = output.view(-1, len(self.dict))
            self.seen = True
        else:
            a.append(self.dict[targets.select(1, 0).item()])
            output, hidden = self.model(targets.select(1,0).view(1, 1), self.hidden)
            self.hidden = self.repackage_hidden(hidden)
            output_flat = output.view(-1, len(self.dict))

        #print(' '.join(a))
        # get probabilites for all words
        probs = F.softmax(output_flat, dim=1).squeeze().cpu()
        probs = probs.tolist()
        dist = self.probs
        for i in range(len(probs)):
            dist[self.dict[i]] = probs[i]

        return dist


if __name__ == '__main__':
    parser = setup_args()
    parser.add_argument('-vme', '--validation-max-exs', type=int, default=-1)
    parser.set_params(
        model='projects.convai2.pgoel.eval_ppl:LanguageModelEntry',
        model_file='/Users/pgoel/Git/ParlAI/projects/convai2/pgoel/mylanguagemodel',
        dict_file='/Users/pgoel/Git/ParlAI/projects/convai2/pgoel/mylanguagemodel.dict',
        batchsize=1,
    )
    opt = parser.parse_args()
    #opt['model_type'] = 'language_model'
    #fnames = ['model', 'model.dict', 'model.opt']
    #download_models(opt, fnames, 'convai2', version='v2.0',
    #                use_model_type=True)
    eval_ppl(opt)
