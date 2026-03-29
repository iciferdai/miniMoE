from modelMgmt import *
from Main_Train import pre_init

def main_eval_manual(cpt_name):
    my_mgmt = pre_init(False)
    print('init evaluate...')
    my_mgmt.init_eval()
    print('load checkpoint...')
    my_mgmt.load_checkpoint(cpt_name, True)
    time.sleep(0.01)
    input_t = input("\nPress send input: ")
    for i in range(5):
        my_mgmt.predict_manual(input_t,'TOP_K',1.2, 5, True)
    my_mgmt.predict_manual(input_t,'BEST', 1.0, 0, True)

if __name__ == '__main__':
    main_eval_manual('overfit/CheckPoint_Ep82219_0.1113.pth')
    #main_eval_manual('overfit2/CheckPoint_Ep108610_0.8723.pth')
    #main_eval_manual('CheckPoint_Ep151168_1.8144.pth')
