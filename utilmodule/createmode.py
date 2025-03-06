 


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))



from PAMIL.models.basemodel import BasedMILTransformer as BaseMILT
from PAMIL.models.DPSF import PPO,Memory
from PAMIL.models.classmodel import ClassMultiMILTransformer as ClassMMILT
from PAMIL.models.SFFR import FusionHistoryFeatures
from PAMIL.utilmodule.utils import make_parse



def create_model(args):
    
    basedmodel = BaseMILT(args).cuda()
    ppo = PPO(args.feature_dim,args.state_dim, args.policy_hidden_dim, args.policy_conv,
                        action_std=args.action_std,
                        lr=args.ppo_lr,
                        gamma=args.ppo_gamma,
                        K_epochs=args.K_epochs,
                        action_size=args.action_size
                        )
    FusionHisF = FusionHistoryFeatures(args.feature_dim,args.state_dim).cuda() #feature_dim, state_dim

    classifymodel = ClassMMILT(args).cuda()
    memory = Memory()
    
    assert basedmodel is not None, "creating model failed. "
    print(f"basedmodel Total params: {sum(p.numel() for p in basedmodel.parameters()) / 1e6:.2f}M")
    print(f"classifymodel Total params: {sum(p.numel() for p in classifymodel.parameters()) / 1e6:.2f}M")
    
    return basedmodel,ppo,classifymodel,memory ,FusionHisF

if __name__ == "__mian__":
    
    args = make_parse()
    create_model(args)
