from options.base_option import BaseOptions

class EvalT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--which_epoch', type=str, default="all", help='Name of this trial')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

        self.parser.add_argument('--ext', type=str, default='default', help='Batch size of pose discriminator')
        self.parser.add_argument("--num_batch", default=2, type=int,
                                 help="Number of repetitions, per sample (text prompt/action)")
        self.parser.add_argument("--repeat_times", default=3, type=int,
                                 help="Number of repetitions, per sample (text prompt/action)")
        self.parser.add_argument("--cond_scale", default=4, type=float,
                                 help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
        self.parser.add_argument("--temperature", default=1., type=float,
                                 help="Sampling Temperature.")
        self.parser.add_argument("--topkr", default=0.9, type=float,
                                 help="Filter out percentil low prop entries.")
        self.parser.add_argument("--time_steps", default=18, type=int,
                                 help="Mask Generate steps.")
        self.parser.add_argument("--seed", default=3407, type=int)

        self.parser.add_argument('--gumbel_sample', action="store_true", help='Training iterations')
        self.parser.add_argument('--use_res_model', action="store_true", help='Training iterations')

        self.parser.add_argument('--res_name', type=str, default='pres_vq6ns_nres6', help='Batch size of pose discriminator')
        self.parser.add_argument('--text_path', type=str, default="./eval_results/text_prompt.txt", help='Name of this trial')


        self.parser.add_argument('-msec', '--mask_edit_section', nargs='*', type=str, help='Indicate sections for editing, use comma to separate the start and end of a section'
                                 'type int will specify the token frame, type float will specify the ratio of seq_len')
        self.parser.add_argument('--seq_len', type=int, help='Indicate length of the sequence and motion, for editalpha only. by frame')
        self.parser.add_argument('-oms', '--og_motion_start', nargs='*', type=int, help='Specify the frame to put the original motion, for editalpha only. by frame')
        self.parser.add_argument('--text_prompt', default='', type=str, help="A text prompt to be generated. If empty, will take text prompts from dataset.")


        self.is_train = False