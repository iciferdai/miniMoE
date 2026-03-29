from modelMgmt import *


def pre_init(need_data=True, sft_data=False):
    logging.info('init model...')
    my_model = MiniMoE()
    my_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if need_data:
        print('Preparing data...')
        if not sft_data:
            train_dataloader = process_data()
        else:
            train_dataloader = process_sft_data()
    else:
        train_dataloader = None
    logging.info('init ModelManagement...')
    my_mgmt = ModelManagement(my_model, train_dataloader, my_dev)
    return my_mgmt


def main_train(steps, fp):
    my_mgmt = pre_init()
    logging.info('init train...')
    # 先权重，后.to
    my_mgmt.init_weights()
    my_mgmt.init_train(fp)
    logging.info('Start train...')
    my_mgmt.train_steps(steps, fp)
    my_mgmt.save_state()
    my_mgmt.show_dashboard()


def load_train(steps, cpt_name, sts_name, fp):
    my_mgmt = pre_init()
    print('init train...')
    my_mgmt.init_train(fp)
    my_mgmt.load_checkpoint(cpt_name)
    my_mgmt.load_state(sts_name)
    print('Start train...')
    my_mgmt.train_steps(steps, fp)
    my_mgmt.save_state()
    my_mgmt.show_dashboard()


def sft_train(cpt_name, steps, fp):
    my_mgmt = pre_init(True, True)
    print('init train...')
    my_mgmt.init_sft_train(cpt_name)
    print('Start train...')
    my_mgmt.train_sft_steps(steps, fp)
    my_mgmt.save_state()
    my_mgmt.show_dashboard()


def load_sft_train(cpt_name, sts_name, steps, fp):
    my_mgmt = pre_init(True, True)
    print('init train...')
    my_mgmt.init_sft_train(cpt_name)
    my_mgmt.load_state(sts_name)
    print('Start train...')
    my_mgmt.train_sft_steps(steps, fp, True)
    my_mgmt.save_state()
    my_mgmt.show_dashboard()


def check_status(cpt_name, sts_name):
    my_mgmt = pre_init(False)
    logging.info('load status of best_test...')
    my_mgmt.init_train()
    my_mgmt.init_dashboard()
    my_mgmt.load_checkpoint(cpt_name)
    my_mgmt.load_state(sts_name)
    my_mgmt.progress_info(True)
    my_mgmt.show_dashboard()


if __name__ == '__main__':
    #main_train(20000, True)
    #load_train(10000,'CheckPoint_Ep122157_4.0850.pth','State_Ep122157_3.8278.pkl', True)
    #load_train(10000, 'CheckPoint_Ep132157_4.0001.pth', 'State_Ep132157_3.7724.pkl', True, sft_data=True)
    #sft_train('CheckPoint_Ep10000_3.1829.pth',10000, True)
    load_sft_train('CheckPoint_Ep151168_1.8144.pth','State_Ep151168_1.7252.pkl',3000, True)
    #check_status('CheckPoint_Ep60000_1.4640.pth','State_Ep60000_1.4241.pkl')