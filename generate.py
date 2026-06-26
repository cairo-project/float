import os, datetime
from dataclasses import fields

from float.inference import InferenceAgent, InferenceConfig
from options.base_options import BaseOptions


class InferenceOptions(BaseOptions):
	def initialize(self, parser):
		super().initialize(parser)
		parser.add_argument("--ref_path", default=None, type=str, help='ref image path')
		parser.add_argument('--aud_path', default=None, type=str, help='audio path')
		parser.add_argument('--emo', default=None, type=str, help='emotion',
				choices=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
		parser.add_argument('--no_crop', action='store_true', help='skip face crop')
		parser.add_argument('--res_video_path', default=None, type=str, help='output video path')
		parser.add_argument('--ckpt_path', default="./checkpoints/float.pth", type=str, help='checkpoint path')
		parser.add_argument('--res_dir', default="./results", type=str, help='result dir')
		return parser


def main():
	opt = InferenceOptions().parse()

	config_keys = {f.name for f in fields(InferenceConfig)}
	config = InferenceConfig(**{k: v for k, v in vars(opt).items() if k in config_keys})
	config.rank = 0
	config.ngpus = 1

	agent = InferenceAgent(config)
	os.makedirs(opt.res_dir, exist_ok=True)

	ref_path = opt.ref_path
	aud_path = opt.aud_path

	if opt.res_video_path is None:
		video_name = os.path.splitext(os.path.basename(ref_path))[0]
		audio_name = os.path.splitext(os.path.basename(aud_path))[0]
		call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
		res_video_path = os.path.join(
			opt.res_dir,
			f"{call_time}-{video_name}-{audio_name}-nfe{opt.nfe}-seed{opt.seed}"
			f"-acfg{opt.a_cfg_scale}-ecfg{opt.e_cfg_scale}-{opt.emo}.mp4"
		)
	else:
		res_video_path = opt.res_video_path

	agent.run_inference(
		res_video_path,
		ref_path,
		aud_path,
		a_cfg_scale=opt.a_cfg_scale,
		r_cfg_scale=opt.r_cfg_scale,
		e_cfg_scale=opt.e_cfg_scale,
		emo=opt.emo,
		nfe=opt.nfe,
		no_crop=opt.no_crop,
		seed=opt.seed,
	)


if __name__ == '__main__':
	main()
