import os
# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

########################################################################################################
#
# Use '/' in model path, instead of '\'. Use ctx4096 models if you need long ctx.
#
# fp16 = good for GPU (!!! DOES NOT support CPU !!!)
# fp32 = good for CPU
# bf16 = worse accuracy, supports CPU
# xxxi8 (example: fp16i8, fp32i8) = xxx with int8 quantization to save 50% VRAM/RAM, slower, slightly less accuracy
#
# We consider [ln_out+head] to be an extra layer, so L12-D768 (169M) has "13" layers, L24-D2048 (1.5B) has "25" layers, etc.
# Strategy Examples: (device = cpu/cuda/cuda:0/cuda:1/...)
# 'cpu fp32' = all layers cpu fp32
# 'cuda fp16' = all layers cuda fp16
# 'cuda fp16i8' = all layers cuda fp16 with int8 quantization
# 'cuda fp16i8 *10 -> cpu fp32' = first 10 layers cuda fp16i8, then cpu fp32 (increase 10 for better speed)
# 'cuda:0 fp16 *10 -> cuda:1 fp16 *8 -> cpu fp32' = first 10 layers cuda:0 fp16, then 8 layers cuda:1 fp16, then cpu fp32
#
# Basic Strategy Guide: (fp16i8 works for any GPU)
# 100% VRAM = 'cuda fp16'                   # all layers cuda fp16
#  98% VRAM = 'cuda fp16i8 *1 -> cuda fp16' # first 1 layer  cuda fp16i8, then cuda fp16
#  96% VRAM = 'cuda fp16i8 *2 -> cuda fp16' # first 2 layers cuda fp16i8, then cuda fp16
#  94% VRAM = 'cuda fp16i8 *3 -> cuda fp16' # first 3 layers cuda fp16i8, then cuda fp16
#  ...
#  50% VRAM = 'cuda fp16i8'                 # all layers cuda fp16i8
#  48% VRAM = 'cuda fp16i8 -> cpu fp32 *1'  # most layers cuda fp16i8, last 1 layer  cpu fp32
#  46% VRAM = 'cuda fp16i8 -> cpu fp32 *2'  # most layers cuda fp16i8, last 2 layers cpu fp32
#  44% VRAM = 'cuda fp16i8 -> cpu fp32 *3'  # most layers cuda fp16i8, last 3 layers cpu fp32
#  ...
#   0% VRAM = 'cpu fp32'                    # all layers cpu fp32
#
# Use '+' for STREAM mode, which can save VRAM too, and it is sometimes faster
# 'cuda fp16i8 *10+' = first 10 layers cuda fp16i8, then fp16i8 stream the rest to it (increase 10 for better speed)
#
# Extreme STREAM: 3G VRAM is enough to run RWKV 14B (slow. will be faster in future)
# 'cuda fp16i8 *0+ -> cpu fp32 *1' = stream all layers cuda fp16i8, last 1 layer [ln_out+head] cpu fp32
#
# ########################################################################################################

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

# download models: https://huggingface.co/BlinkDL
model = RWKV(model='/Users/yueyulin/Downloads/RWKV-5-World-3B-v2-20231113-ctx4096.pth', strategy='cpu fp32')
print(model.args)
print(model.version)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV

ctx = "User:根据以下新闻，写出新闻标题：网易科技讯1月20日消息，今日美团创始人CEO王兴给全体员工发了一封内部信。内部信中提到：公司联合创始人、S-team成员、高级副总裁王慧文将于今年12月退出公司具体管理事务。后续王慧文将继续担任公司董事，并任美团终身荣誉顾问。美团联合创始人、高级副总裁王慧文随后，美团发布了多项组织调整与任命公告，宣布成立美团平台，负责美团App、设计及市场营销相关工作，由副总裁李树斌负责，向王慧文汇报；原用户平台点评App部更名为点评事业部，负责人为黄海不变，继续向王慧文汇报；成立AI平台和新的基础研发平台，负责强化相关技术能力，更好的服务以及赋能好公司的各个业务。内部信中还宣布美团将增补副总裁郭庆、副总裁李树斌为S-team成员。郭庆于2014年3月加入美团，目前主要负责美团酒店、门票度假、民宿等业务；李树斌在2019年12月刚刚加入美团，在新的组织架构中担任美团平台负责人。与此同时，S-team成员、高级副总裁刘琳（Elaine）因个人及家庭原因，将于今年转任公司高级顾问，后续Elaine将继续投入时间精力，助力公司发展尤其是人力资源体系建设。在王慧文发布的内部信中，他提到这是适合交棒的时间点。王慧文称，一直以来都不能很好的处理工作与家庭、健康，业务经营与个人兴趣之间的关系，也担心人生被惯性主导，怠于熟悉的环境而错过了不同的精彩。“面向互联网‘下半场’和未来十年的准备工作已经初见成效，在兴哥和新S-team的带领下美团一定会不断突破，创造更多社会价值，这是适合交棒的时间点。”他强调，“2020年我会和大家并肩作战，全力以赴推动美团发展，确保各项工作有序交接。”（一橙）王慧文个人经历2003年底从中科院声学所退学创业2005年底与王兴，赖斌强一起创办校内网2006年校内网获得亚马逊前首席科学家韦斯岸的天使投资2006年10月，千橡互动集团收购校内网2008年6月，创建北京乐新创展科技有限公司2010年4月，推出淘房网。淘房网是一个二手房网站，允许买家对中介进行点评，很像淘宝网2011年，放弃淘房网，重新回到王兴麾下，加入美团网。担任副总裁职务，负责美团网的市场和产品相关工作。培养和招募了一大批人才，为美团网的市场和产品发展奠定了坚实的基础2019年10月，王慧文卸任美团打车运营主体上海路团科技有限公司的法定代表人。2019年12月，上海三快科技有限公司发生一系列变更，王慧文退出法定代表人。2019年12月20日，北京摩拜科技有限公司发生工商变更，美团联合创始人、高级副总裁王慧文卸任北京摩拜科技有限公司法定代表人、执行董事和经理一职。以下为王兴和王慧文的内部信：不忘初心薪火相传——公司启动“领导梯队培养计划”各位同事：我们即将迎来美团十岁生日，比这更重要的是，我们将迎来公司新的十年。“人是美团最重要的资产”，面向下一个十年，我们要继续践行好“帮大家吃得更好，生活更好”的使命，就一定要让一批又一批新的管理者成长起来。为此，公司决定启动“领导梯队培养计划”，推动公司人才盘点、轮岗锻炼、继任计划等一系列工作有序开展，为人才梯队培养提供组织和制度保障，为大家和公司共同发展创造更好的条件。经慎重考虑，公司决定增补副总裁郭庆、副总裁李树斌为S-team（Seniorteam）成员。未来我们将基于公司长期发展需要，持续加强S-team自身建设。同时，联合创始人、S-team成员、高级副总裁王慧文（老王）将于今年12月退出公司具体管理事务，开启人生新的篇章。今年内，老王将一如既往全力推动公司业务发展、提升组织能力，并将投入更多精力到人才梯队培养上；后续老王将继续担任公司董事，并任美团终身荣誉顾问、“互联网+大学”特别讲师，助力公司战略规划、组织传承和人才发展。S-team成员、高级副总裁刘琳（Elaine）因个人及家庭原因，将于今年转任公司高级顾问，后续Elaine将继续投入时间精力，助力公司发展尤其是人力资源体系建设。一支队伍，越是有远大的使命，越需要久久为功、薪火相传。十年树木，百年树人，S-team推动自身变化正是着眼于此。感谢老王和Elaine在公司发展过程中做出的卓越贡献，期待他们在接下来的一段时间里为“领导梯队培养计划”的起步投入更多精力，期待郭庆和树斌把自己多年的实践、思考带入S-team，为公司发展作出更大贡献，也期待未来更多的管理者成长起来！更好的十年，让我们既往不恋，纵情向前！王兴2020.1.20王慧文回复邮件今年3月4日是美团十周年，在这十年移动互联网的伟大浪潮中，我们的业务快速发展，创造的社会价值越来越大，团队茁壮成长，各项基础建设稳步推进。我有信心说面向互联网“下半场”和未来十年的准备工作已经初见成效，在兴哥和新S-team的带领下美团一定会不断突破，创造更多社会价值，这是适合交棒的时间点。一直以来我都不能很好的处理工作与家庭、健康的关系；也处理不好业务经营所需要的专注精进与个人散乱不稳定的兴趣之间的关系；不热爱管理却又不得不做管理的痛苦也与日俱增；我也一直担心人生被惯性主导，怠于熟悉的环境而错过了不同的精彩。今年12月18日是我在美团的十周年，这十年激烈精彩，不负年华。届时我将退休，换一个人生轨道和生活方式。感谢伟大的时代，我生于1978年，是改革开放的同龄人；在我开始厌学的时候，大学宿舍通网，因此赶上了互联网最精彩的20年；中国作为全球最大的单一市场，对创业者来说更是得天独厚。我运气实在太好，不宜继续贪天之功，知止不殆。感谢我的家人，承受很多困难，无怨无尽的给我支持和包容。感谢兴哥，跟兴哥同宿舍是我生命中另外一个巨大的运气，兴哥帮我的人生打开了一扇窗，给了我舞台和机会，在每个关键选择里指出了正确的方向。感谢所有同事，不仅给了我巨大的帮助，更给了我精神的力量，尤其包容了我古怪的性格和偏激的语言。2020年我会和大家并肩作战，全力以赴推动美团发展，确保各项工作有序交接。美团十年，豪兴不浅。他日江湖相逢，再当杯酒言欢。老王2020.1.20王兴回复王慧文邮件王慧文是1997年8月底我到清华入学报到时认识的第一个同学，我们宿舍六个人里有两个姓王，他比我大几个月，所以他就成了老王。转眼23年，从清华到校内网，再到美团，老王和我是有共同志趣的同学和室友，是携手创业的搭档和并肩战斗的战友，更是可以思想碰撞、灵魂对话的一生挚友。过去几年老王和我多次讨论过他的退休计划，公司也为此一直在做准备，今天，我们正式和大家沟通这件事，我心里有理解，也有不舍，但更多的是感谢和祝福。美团十年，老王全情投入、贡献卓著。2010年12月老王携团队加入初创时期的美团，胜利会师，帮助公司从“千团大战”中脱颖而出；2013年，老王创始美团外卖，从最初只有两个人开始一手缔造了今天公司的核心支柱；2016年开始老王又多方探索，为公司深入餐饮产业链上下游、构建到店/到家/出行等业务场景打下了坚实基础；最近一年多，老王马不停蹄，大刀阔斧推动用户平台、基础研发、大数据和AI等平台能力建设。作为公司联合创始人、执行董事，老王还持续在公司的资本战略、公共战略、科技战略、组织战略、文化培育和人才发展等一系列关乎美团长远发展的重大议题上贡献他的智慧和力量。美团精神，老王身体力行、堪称典范。回顾老王过去九年多的工作，既有冲锋在前的勇猛，又有安营扎寨的稳健；既有舍我其谁的担当，又有功成不必在我的潇洒；既有“天下兴亡，匹夫有责”的责任感，又有“我们什么都没有，但是我们有兄弟和勇气”的真性情；既长期有耐心地保持战略定力，又坚持时不我待、只争朝夕地忘情投入。我想，老王就是美团人的代表，老王身上展现出的这些闪光点，就是美团精神。回顾过去，是为了更好面向未来。我相信老王在接下来的一年会继续帮助公司夯实基本功，在领导梯队培养、组织能力建设方面给大家更多帮助；期待老王未来换一种角色后，以另一种方式继续助力公司战略规划、组织传承和人才发展。我相信将来老王会书写更精彩的人生篇章。大江大河，皆向大海。今后无论我们在什么样的轨道，用什么样的方式，都会继续求真务实、为客户和社会创造价值、不负时代。感谢老王，祝福老王！无问西东，纵情向前！王兴2020.1.20本文来源：网易科技报道责任编辑：乔俊婧_NBJ11279\nAssistant: "
print(ctx, end='')

def my_print(s):
    print(s, end='', flush=True)

# For alpha_frequency and alpha_presence, see "Frequency and presence penalties":
# https://platform.openai.com/docs/api-reference/parameter-details

args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     alpha_decay = 0.996, # gradually decay the penalty
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

pipeline.generate(ctx, token_count=100, args=args, callback=my_print)
print('\n')

out, state = model.forward([187, 510, 1563, 310, 247], None)
print(out.detach().cpu().numpy())                   # get logits
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())                   # same result as above
print('\n')