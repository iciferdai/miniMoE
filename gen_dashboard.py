import logging
import pickle
import matplotlib.pyplot as plt

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ion
plt.switch_backend('TkAgg')


def load_state(state_name=''):
    if not state_name:
        print('No state provided.')
        return
    state_path = './saves/' + state_name
    try:
        with open(state_path, 'rb') as f:
            manager_state = pickle.load(f)
            train_loss_list = manager_state['train_loss_list']
            best_checkpoints = manager_state['best_checkpoints']
            logging.info(f"State: {state_path} Loaded")
            return train_loss_list, best_checkpoints
    except Exception as e:
        logging.error(f"load_state Error: {e}", exc_info=True)
        exit(1)


def show_dashboard(d_list):
    plt.ion()
    fig, ax = plt.subplots(figsize=(20, 6), num="Loss Dashboard")
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Dashboard')
    ax.grid(alpha=0.01)

    train_line, = ax.plot(range(1, len(d_list)+1), d_list, label='Train Loss')

    ax.legend()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.ioff()


def calculate_datas():
    file_path=''
    with open(file_path, "rb") as f:
        d = pickle.load(f)
    return d


if __name__ == '__main__':
    #pre_loss, _ = load_state('pre_train/State_Ep132157_3.7724.pkl')
    pre_loss, _ = load_state('1.5B_20000steps_bak/State_Ep20000_7.1483.pkl')
    #sft_loss, _ = load_state('overfit/State_Ep82219_0.1026.pkl')
    #sft_loss, _ = load_state('overfit2/State_Ep108610_0.7910.pkl')
    #sft_loss, _ = load_state('State_Ep151168_1.7252.pkl')
    #pre_loss.extend(sft_loss)
    show_dashboard(pre_loss)

    plt.show()