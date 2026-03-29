import logging
import json
from myTrans.base_params import *
import pprint
import re
import pickle
import random
import concurrent.futures
from data_dict import *
import sys
from multiprocessing import Pool
import array


def load_tmp_data(file_path):
    with open(file_path, "rb") as f:
        d = pickle.load(f)
    return d


def save_tmp_data(file_path, d):
    with open(file_path, "wb") as f:
        pickle.dump(d, f)
    logging.info(f"data {type(d)} saved at {file_path}")


def save_txt_data(file_path, d):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for question, answer in d:
                line = f"{question} | {answer}\n"
                f.write(line)
        logging.info(f"save_txt_data data {type(d)} saved at {file_path}")
    except Exception as e:
        print(f"Error: {e}")


# 预计算str_list的词频字典（O(n)一次计算，替代多次O(n)的count）
def get_freq_dict(lst):
    freq = {}
    for item in lst:
        freq[item] = freq.get(item, 0) + 1
    return freq

def process_first_vocab(d_l):
    full_str_list = []
    pos_list = []
    full_set = set()
    for batch_idx, (pos, content) in enumerate(d_l):
        full_str_list.extend(content)
        pos_list.extend(pos)
        # BS(128) * 100
        if batch_idx%100 == 0:
            full_str = ''.join(full_str_list)
            logging.info(f"batch_idx: {batch_idx}, data num: {len(full_str)}, pos: {len(pos_list)}")
            invalid_pattern = r'[^\u4e00-\u9fa5A-Za-z0-9。！？；，：、…—”“（）《》·「」『』【】,.!?;:(){}[\]<>"\'’‘-]'
            allowed_pattern = r'[\u4e00-\u9fa5A-Za-z0-9。！？；，：、…—”“（）《》·「」『』【】·,.!?;:(){}[\]<>"\'’‘-]'
            invalid_t = re.findall(invalid_pattern, full_str)
            allowed_t = re.findall(allowed_pattern, full_str)
            if invalid_t: print(f'find invalid_t {set(invalid_t)}')
            str_list = list(allowed_t)
            str_set = set(str_list)
            full_set.update(str_set)
            save_tmp_data(f'./local_datasets/str_list_{batch_idx}.pkl', str_list)
            logging.info(f"full_set len latest: {len(full_set)}, str_list_{batch_idx}(len:{len(str_list)}) saved ")
            full_str_list = []
            pos_list = []

    save_tmp_data('./local_datasets/full_set.pkl', full_set)
    print(f'final full_set len: {len(full_set)}, saved')
    print(full_set)

def filter_by_threshold(threshold=2850):
    logging.info('loading file')
    voc_set = load_tmp_data('./local_datasets/full_set.pkl')
    str_list = load_tmp_data('./local_datasets/str_list_0.pkl')
    str_list_freq = get_freq_dict(str_list)
    ok_dict=dict()
    nok_dict=dict()
    logging.info('deal first time')
    for one_t in voc_set:
        count_num = str_list_freq.get(one_t, 0)
        if count_num >= threshold:
            ok_dict[one_t] = (count_num, 0)
        else:
            nok_dict[one_t] = count_num
    logging.info(f'first deal finished: {len(ok_dict)}|{len(nok_dict)}')

    for idx in range(1, 190):
        data_list_file = f'./local_datasets/str_list_{idx}00.pkl'
        logging.info(f'deal file: {data_list_file}')
        str_list = load_tmp_data(data_list_file)
        str_list_freq = get_freq_dict(str_list)
        to_delete = []
        for i, (k, v) in enumerate(nok_dict.items()):
            if i%2000 == 0: logging.info(f'deal[{i}/{len(nok_dict)}] k: {k}')
            add_num = str_list_freq.get(k, 0)
            count_num = add_num + v
            if count_num >= threshold:
                ok_dict[k] = (count_num, idx)
                to_delete.append(k)
            else:
                nok_dict[k] = count_num

        for k in to_delete: del nok_dict[k]

        logging.info(f'No_{idx} deal finished: {len(ok_dict)}|{len(nok_dict)}, changed {len(to_delete)}')
        save_tmp_data(f'./local_datasets/ok_dict_{idx}.pkl', ok_dict)
        save_tmp_data(f'./local_datasets/nok_dict_{idx}.pkl', nok_dict)

# 极限套2层
def get_next_token(one_str, x, pair_priority):
    tmp_token = (one_str[x], one_str[x+1])
    y = 1
    if tmp_token not in pair_priority:
        tmp_token = one_str[x]
        y = 0
    return tmp_token, y

def sub_get_max_pair(one_str, pair_priority):
    # 统计所有两两相邻的频率
    freq = {}
    # 边界保护简单-3，不差1个字，少大量判断
    for i in range(len(one_str) - 3):
        first_token, y = get_next_token(one_str, i, pair_priority)
        second_token, z = get_next_token(one_str, i+1+y, pair_priority)
        pair = (first_token, second_token)
        if pair in pair_priority: continue
        freq[pair] = freq.get(pair, 0) + 1
    # 选频率最高的对合并
    best_pair = max(freq, key=freq.get)
    #logging.info(f'Best pair: {best_pair}')
    return best_pair

def sub_get_topk_pair(one_str, pair_priority, topk=3):
    # 统计所有两两相邻的频率
    freq = {}
    # 边界保护简单-3，不差1个字，少大量判断
    for i in range(len(one_str) - 3):
        first_token, y = get_next_token(one_str, i, pair_priority)
        second_token, z = get_next_token(one_str, i+1+y, pair_priority)
        pair = (first_token, second_token)
        if pair in pair_priority: continue
        freq[pair] = freq.get(pair, 0) + 1
    #排序，取top
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:topk]

def sub_add_tasks(idx, pair_priority, scale_rate=1):
    tmp_best_pairs = dict()
    data_list_file = f'./cache_data/str_list/str_list_{idx}00.pkl'
    str_list = load_tmp_data(data_list_file)

    one_str = ''.join(str_list)
    # 忽略边界，简化处理，概率删选本身可以保证
    zh_allowed_pattern = r'[\u4e00-\u9fa5]'
    en_allowed_pattern = r'[A-Za-z]'
    num_allowed_pattern = r'[0-9]'
    str_zh = re.findall(zh_allowed_pattern, one_str)
    str_en = re.findall(en_allowed_pattern, one_str)
    str_num = re.findall(num_allowed_pattern, one_str)

    # 中 : 英 : 数字 = 1200:250:50 = 24:5:1
    best_pairs_zh = sub_get_topk_pair(str_zh, pair_priority, 24*scale_rate)
    best_pair_en = sub_get_topk_pair(str_en, pair_priority, 5*scale_rate)
    if scale_rate == 1:
        best_pair_num = sub_get_max_pair(str_num, pair_priority)
    else:
        best_pair_num = sub_get_topk_pair(str_num, pair_priority, scale_rate)
    for pair_zh in best_pairs_zh:
        tmp_best_pairs[pair_zh[0]] = 1
    for pair_en in best_pair_en:
        tmp_best_pairs[pair_en[0]] = 1
    if scale_rate == 1:
        tmp_best_pairs[best_pair_num] = 1
    else:
        for pair_num in best_pair_num:
            tmp_best_pairs[pair_num[0]] = 1

    return tmp_best_pairs

def sub_combine_result(tmp_dict, combine_best_pair, pair_priority):
    for p, count in tmp_dict.items():
        combine_best_pair[p] = combine_best_pair.get(p, 0) + count
        if p not in pair_priority: pair_priority.append(p)

def add_vocab(sample_num=4, top_k=1500):
    logging.info('add_vocab start')
    combine_best_pair = dict()
    pair_priority = []
    scale_rate=1
    while len(combine_best_pair) < top_k:
        # 随机采样
        idx1, idx2, idx3, idx4 = [random.randint(1, 189) for _ in range(4)]
        logging.info(f'id: {idx1}, {idx2}, {idx3}, {idx4}')
        tasks = [(sub_add_tasks, idx1, pair_priority, scale_rate),
                 (sub_add_tasks, idx2, pair_priority, scale_rate),
                 (sub_add_tasks, idx3, pair_priority, scale_rate),
                 (sub_add_tasks, idx4, pair_priority, scale_rate)]

        # 起多线程并行加速：
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 提交任务并记录 future 对象
            sub_task1 = executor.submit(tasks[0][0], tasks[0][1], tasks[0][2], tasks[0][3])
            sub_task2 = executor.submit(tasks[1][0], tasks[1][1], tasks[1][2], tasks[1][3])
            sub_task3 = executor.submit(tasks[2][0], tasks[2][1], tasks[2][2], tasks[1][3])
            sub_task4 = executor.submit(tasks[3][0], tasks[3][1], tasks[3][2], tasks[3][3])

            # 获取结果（会阻塞，直到对应线程完成）
            tmp_best_pairs1 = sub_task1.result()
            tmp_best_pairs2 = sub_task2.result()
            tmp_best_pairs3 = sub_task3.result()
            tmp_best_pairs4 = sub_task4.result()

        # 结束并行部分
        sub_combine_result(tmp_best_pairs1, combine_best_pair, pair_priority)
        sub_combine_result(tmp_best_pairs2, combine_best_pair, pair_priority)
        sub_combine_result(tmp_best_pairs3, combine_best_pair, pair_priority)
        sub_combine_result(tmp_best_pairs4, combine_best_pair, pair_priority)

        show_id1, show_id2, show_id3 = [random.randint(0, len(pair_priority)) for _ in range(3)]
        logging.info(f'Progress: {len(pair_priority)}|{top_k}|{scale_rate}; show case: {pair_priority[show_id1]}|{pair_priority[show_id2]}|{pair_priority[show_id3]}')
        scale_rate *= 2

    print(f'Final pair total: {len(combine_best_pair)}|{len(pair_priority)}, top 10: {pair_priority[:10]}, bottom 10: {pair_priority[-10:-1]}')
    print(combine_best_pair)
    save_tmp_data(f'./cache_data/pairs_top_1500.pkl', pair_priority)
    save_tmp_data(f'./cache_data/combine_best_pair.pkl', combine_best_pair)

def combine_str():
    pair_priority = load_tmp_data('./cache_data/pairs_top_1500.pkl')
    voc_list = []
    print(pair_priority)
    for pair in pair_priority:
        combine_word = ''
        first_str, second_str = pair
        if type(first_str) is tuple:
            pre_str, post_str = first_str
            combine_word = combine_word + pre_str + post_str
        else:
            combine_word += first_str

        if type(second_str) is tuple:
            pre_str, post_str = second_str
            combine_word = combine_word + pre_str + post_str
        else:
            combine_word += second_str

        voc_list.append(combine_word)
    print(voc_list)
    print(len(voc_list))

def combine_voc():
    final_voc_dict=dict()
    id = CUS_START_ID
    voc_set = load_tmp_data('./cache_data/ok_dict_55.pkl')
    pair_priority = ['公司', '一个', '中国', '我们', '可以', '自己', '没有', '投资', '市场', '工作', '基金', '企业', '就是', '进行', '时间', '发展', '这个', '管理', '如果', '问题', '孩子', 'ra', 'ac', 'in', 'er', 'an', '00', '服务', '他们', '什么', 'fr', 'cf', '01', 'on', 'PP', '已经', '因为', '可能', '通过', '关注', '需要', '产品', '时候', '这样', '经济', '不是', '情况', '很多', '方面', '国家', '记者', '技术', '生活', '银行', '目前', '现在', '同时', '信息', '美国', '开始', '教育', '项目', '还是', '微信', '所以', '但是', '这些', '数据', '第一', '成为', '活动', '对于', '表示', '大的', '人员', '自己的', '行业', 'frac', 'acfr', 'en', 'th', 're', 'ng', 'or', '12', '201', '个人', '大学', '交易', '重要', '世界', 'am', 'mp', 'pa', '影响', 'at', 'ar', 'om', 'es', 'st', 'ne', 'nt', '一些', '金融', 'he', '以及', '设计', '人民', '提供', '社会', '也是', '到了', '大家', '都是', '建设', '部分', '学生', '研究', '价格', '作为', '选择', '系统', '要求', '平台', '好的', '主要', '汽车', '使用', '相关', '出现', '文化', '非常', '朋友', '安全', '发现', '一定', '北京', '其他', '增长', '不能', '城市', '方式', '能力', '知道', '还有', '认为', '的时候', '更多', '一次', '学习', '不同', '证券', '了解', '根据', '其中', '专业', '国际', '政府', '一种', '有限', '联网', '中心', '美元', '政策', '互联', '以上', '为了', '那么', '实现', '内容', '业务', '之后', '所有', '上海', '品牌', '机构', '产业', '合作', '资金', '这种', '这一', '有的', '比较', '手机', '或者', '消费', '亿元', '创新', '不会', '能够', '今年', '销售', '发生', 'al', 'nd', 'el', 'ou', 'te', 'le', 'ti', 'ed', 'it', 'et', 'ic', 'ww', 'll', 'AA', 'ing', 'com', 'tt', 'ro', 'is', 'ri', '02', '11', '15', '14', '用户', '资产', '未来', '报告', '其实', '分析', '起来', '一起', 'ht', 'mm', 'ch', '科技', '万元', '环境', '部门', '看到', '不要', '学校', 'ec', '16', '风险', 'to', '发布', '健康', '进入', '你的', '第二', '包括', '网络', '全国', '不过', '如何', '互联网', '基本', '一直', '获得', '标准', '而且', '虽然', '特别', '由于', '增加', '怎么', '规定', '应该', '国内', '关系', '一步', '过程', '单位', '是在', '因此', '一般', '希望', '价值', '组织', '需求', '显示', '支持', '这是', '喜欢', '来说', '功能', '资源', '代表', '开发', '有人', '个月', '达到', '地区', '收入', '甚至', '不仅', '运动', '全球', '创业', '完成', '有限公司', '不断', '提高', '最后', '直接', '现场', '工程', '存在', '超过', '计划', '地方', '容易', '模式', '继续', '建议', '媒体', '他的', '旅游', '生产', '提升', '日本', '基础', '觉得', '注意', '电话', '负责', '持续', '新的', '智能', '是一个', '今天', '成功', '有关', '解决', '必须', '介绍', '原因', '空间', '方法', '集团', '目标', '这里', '电影', '这样的', '作用', '生的', '上市', '具有', '只有', '传统', '参加', '领域', '表现', '行为', '控制', '游戏', '老师', '处理', '公众', '结果', '一样', '两个', '改革', '会议', '自然', '经营', '来看', '有效', '交通', '每天', '帮助', '小时', '艺术', '关于', '责任', '医院', '导致', '参与', '比赛', '第三', '重点', '一下', '实际', '各种', '之前', '形成', '董事', '要的', '除了', '也不', '股份', '点击', '左右', '股票', '分钟', '保护', '只是', '历史', '比如', '水平', '操作', '综合', '条件', '考试', '之间', '不少', '科学', '是否', '任何', '家庭', '联系', '采用', '出了', '的话', '开展', '质量', '机会', '一点', '知识', '随着', '保险', '成了', '保持', '客户', 'oo', 'sp', 'as', 'sh', 'ID', 'nb', 'SU', 'ss', 'li', 'UV', 'ion', 'AC', 'lo', 'bs', 'ns', 'PS', 'MA', 'tp', 'la', 'ol', 'ce', 'id', 'of', 'os', 'QQ', 'PE', 'SS', 'ow', 'iv', 'AM', 'us', 'ad', 'ab', 'CC', 'www', 'ag', 'ta', 'PA', 'ai', 'rs', '13', '22', '19', '03', '23', '32', '52', '05', '地产', '先生', '调整', '成本', '按照', '我的', '变化', '正在', '股东', '以下', '网站', '支付', '信号', 'AB', 'BA', 'BB', 'pp', 'nc', 'and', 'IP', 'CD', 'http', 'ma', 'ttp', 'em', 'PI', '人民币', '然后', '体验', '感觉', '事情', '收益', '电视', 'ea', 'ig', '资本', '实施', '更加', '学院', '规模', '精神', '阅读', 'im', 'ty', 'tion', '发行', '投资者', '进一步', '这么', '之一', '分别', '效果', '具体', '无法', '申请', '明显', '准备', '车型', '才能', '父母', '利用', '经过', '以来', '报名', '债券', '建立', '的情况', '结构', '就会', '领导', '购买', '制度', '完全', '来源', '利润', '合同', '积极', '决定', '竞争', '二维码', '告诉', '出来', '接受', '能源', '不得', '消息', '当时', '简单', '造成', '并不', '期间', '指数', '妈妈', '最高', '最终', '免费', '方案', '加强', '时代', '监管', '战略', '得到', '调查', '当然', '的问题', '优势', '身体', '真正', '新闻', '阶段', '来自', '方向', '一家', '东西', '消费者', '作品', '年度', '报道', '长按', '法律', '关键', '一位', '贷款', '材料', '拥有', '计算', '带来', '专家', '不到', '人们', '压力', '正式', '全面', '电子', '成绩', '受到', '公里', '系列', '加入', '我们的', '只要', '中央', '有些', '设备', '都有', '深圳', '都会', '医疗', '过去', '保障', '坚持', '动力', '来了', '另外', '并且', '比例', '推进', '推荐', '最大', '越来', '应用', '驾驶', '商业', '公众号', '为什么', '一年', '核心', '意见', '生态', '长期', '范围', '去年', '也有', '整体', '如此', '打造', '上涨', '分享', '规划', '费用', '共同', '经验', '严重', '多人', '重大', '说明', '公布', '家长', '属于', '减少', '文章', '不可', '结合', '体育', '区域', '考虑', '努力', '整个', '化的', '下降', '本次', '全部', '执行', '会有', '治疗', '宝宝', '附近', '公告', '识别', '产生', '人才', '故事', '车辆', '年轻', '此外', '下来', '一天', '普通', '女人', '保证', '制造', '人士', '那些', '改变', '体系', '照片', '符合', '许多', '培训', '预计', '我国', '建筑', '快速', '过程中', '目的', '取得', '团队', '房地产', '及时', '真的', '行情', '小编', '检查', '行动', '运营', '以后', '降低', '联合', '促进', '正常', '份额', '融资', '办法', '也会', '食品', '主任', '趋势', '自动', '大会', '事业', '配置', '丰富', '咨询', '尤其', '大量', '稳定', '而是', '交流', '位置', '原油', '最近', '推动', '移动', '年来', '然而', '明确', '即可', '同比', '独立', '针对', '视频', '现代', '社区', '经常', '面对', '季度', '无论', '成长', '状态', '作者', '程度', '居民', '患者', '确定', '基金管理', '成立', '适合', '半年', '委员', '此次', '因素', '情况下', '工业', '儿童', '事件', '网友', '一定要', '高速', '职业', '完善', '平均', '相比', '医生', '十分', '指导', '信用', '理财', '机制', '动机', '多少', '相对', '英国', '连续', '值得', '商品', '当地', '预期', '生命', '音乐', '香港', '相信', '公开', '结束', '群众', '卫生', '行政', '数量', '习惯', '设施', '女性', '经理', '经历', '测试', '协议', '首先', '市民', '发动', '提出', '近日', '升级', '皮肤', '支撑', '运行', '通知', '满足', '办公', '特色', '心理', '突破', '月份', '规范', 'AT', 'LE', 'ay', 'op', 'LL', 'ED', 'CE', 'ew', 'ts', 'TT', 'od', 'hi', 'ia', 'me', 'the', 'ep', 'nbsp', 'ly', 'one', 'bsp', 'ok', 'rd', 'rb', 'IT', 'APP', 'ry', 'ang', 'GD', 'VR', 'og', 'AN', 'ur', 'you', 'se', 'oc', 'SUV', 'PO', 'OO', 'mmmm', 'AR', 'eb', 'ck', 'if', 'ter', 'OS', 'ei', 'ee', 'hon', 'rt', 'Ph', 'hu', 'ov', 'ver', 'GG', 'si', 'ST', 'hin', 'iP', 've', 'ld', 'No', 'ey', 'tu', 'ev', 'CO', 'PG', 'ET', 'ent', 'TC', 'AAAA', 'ati', 'DP', 'EC', 'eS', 'ao', 'ers', 'ip', 'ml', 'ob', '0000', '100', '04', '016', '18', '17', '200', '101', '42', '06', '08', '34', '62', '014', '24', '25', '最新', '报价', '人口', '情报', '生猪', '开放', '面积', '会计', '毕业', '微博', '女士', '电动', '员工', '本基金', '自由', '考生', '不知道', '理解', '农业', '幸福', '时尚', '感受', '广告', '享受', '资料', '欢迎', '韩国', '再次', '一条', '回复', 'DD', 'EO', 'av', 'ef', 'ni', 'sc', 'IA', 'The', 'for', 'TA', 'SE', 'Ch', 'han', 'ha', 'cn', '72', '015', '07', '酒店', '教师', '营销', '宣传', '男人', '黄金', '股权', '充分', '指标', '别人', '力量', '电商', '吸引', '货币', '大多', '机关', '看看', '小学', '公共', '如今', '主题', '登记', 'ex', 'tm', 'pww', 'de', 'AS', 'RV', 'rg', 'il', 'da', '33', '业绩', '财务', '广州', '公安', '用于', '网上', '的投资', '议案', '长的', '下午', '成交', '集中', '引起', '有所', '供应', '几乎', 'ote', 'PC', 'TV', 'TS', 'no', 'EI', 'MM', 'ity', '45', '越来越', '内部', '优秀', '招聘', '组合', '人生', '一切', '食物', '万万', '就可以', '发动机', '办理', '资格', '金额', '同样', '土地', '身份', '不错', '依然', '官方', '做好', '意识', '优惠', '对方', '搭配', '位于', '处于', '相当', '多数', '意义', '鸡蛋', '避免', '财政', '会在', '证明', '程序', '证券投资', '负责人', '利率', '注册', '图片', '指出', '今日', '措施', '岗位', '工作人员', '找到', '启动', '几个', '形式', '报告期', '风格', '合理', '后来', '三个', '安排', '节目', '同学', '每个', '投入', '利益', '培养', '编辑', '发挥', '投资基金', '不好', '农村', '也可以', '最好', '答案', '广东', '往往', '任务', '更是', '近期', '一场', '地址', '委员会', '曾经', '速度', '统计', '监督', '涉及', '真实', '截至', '严格', '等等', '就能', '打开', '制作', '事实', '娱乐', '商务', '儿子', '一段', '上述', '工具', '布局', '这次', '山东', '策略', '有着', '上市公司', '下面', '会上', '协会', '空气', '肯定', '民警', '平方', '加上', '展示', '游客', '昨日', '政治', '推广', '疾病', '还要', '法院', '逐渐', '采取', '上升', '大幅', '创造', '不用', '双方', '不足', '变得', '训练', '统一', '日前', '主动', '大型', '愿意', '在中国', '浙江', '男子', '一方面', '现象', '机器', '干部', '天然', '道路', '那个', '改善', '上午', '课程', '放在', '自身', '报考', '根本', '提醒', '多个', '让你', '精彩', '困难', '首次', '估值', '不再', '实在', '软件', '原则', '从而', '资讯', '确保', '收购', '将会', '即使', '到底', '认识', '在于', '教学', '母亲', '渠道', '确认', '推出', '最佳', '之外', '参考', '并没有', '加快', '不管', '设置', '方便', '明星', '女儿', '湖南', '天津', '管理人', '研发', '一款', '成员', '添加', '一路', '深入', '亿美元', '营养', '美丽', '特点', '文件', '分配', '正确', '书记', '经典', '意味', '天气', '民族', '里面', '募集', '距离', '本身', '美食', '现实', '尽管', '兴趣', '随后', '查看', '高级', '事项', '数字', '很难', '想要', '一个人', '老板', '将在', '增强', '欧洲', '有可能', '排名', '披露', '挑战', '违法', '采访', '国人', '营业', '新能源', '足球', '德国', '就业', '转型', '搜狐', '世纪', '苹果', '举办', '大部分', '于是', '老人', '落实', '持有', '地点', '记录', '期内', '就要', '重新', '角度', '房子', '很大', '日常', '最低', '面临', '优质', '下方', '案件', '油价', '智慧', '重庆', '下跌', '年年', '迅速', '亚洲', '每年', '杭州', '物质', '特殊', '现金', '成都', '犯罪', '宣布', '可能会', '年龄', '观察', '一旦', '公园', '遇到', '幼儿', '来到', '突然', '环保', '本文', '概念', '评价', '绝对', '完美', '能在', '在线', '形象', '主义', '账户', '相应', '中华', '给予', '理论', '提前', '一件', '中学', '当前', '伙伴', '托管', '期待', '观众', '旅行', '贸易', '品质', '关联', '仅仅', '学会', '召开', '动作', '舒适', '效率', '所谓', '确实', '做法', '对象', '大众', '如下', '父亲', '状况', '颜色', '至少', '尤其是', '在一起', '拍摄', '情绪', '掌握', '一线', '总是', '背景', '南京', '人数', '就像', '物流', '昨天', '仍然', '衣服', '动物', '加大', '一张', '似乎', '人类', '举行', '污染', '对此', '一份', '转载', '年前', '快乐', '年代', '限制', '理念', '做到', '判断', '实力', '此前', '提示', '沟通', '从事', '海外', '库存', '终于', '面试', '业内', '思想', '高度', '短期', '鼓励', '具备', '也就是', '文明', '很多人', '性能', '景区', '爸爸', '证监', '实验', '充满', '第四', '家里', '晚上', '走势', '还在', '十年', '获取', '可是', '周期', '河南', '英语', '表达', '使得', '覆盖', '轻松', '四川', '联储', '难以', '各类', '高端', '家人', '江苏', '人体', '强调', '环节', '基地', '爱情', '以前', '法规', '追求', '造型', '永远', '哪些', '事故', '住房', '司机', '还能', '解释', '流动', '类型', '女孩', '评估', '小区', '时期', '力度', '著名', '简称', '女子', '低于', '态度', '多年', '知名', '施工', '较大', '大多数', '定位', '多次', '累计', '依法', '这款', '工资', '即将', '装修', '路上', '进口', '而在', '日起', '规则', '大数据', '类似', '详细', '董事会', '角色', '身边', '公务', '应当', '股市', '航空', '分为', '补贴', '正是', '盈利', '台湾', '刺激', '制定', '教授', '充电', '感情', '探索', '销量', '总经理', '损失', '行车', '技能', '中小', '座椅', '动车', '结婚', '高校', '检测', '期末', '呈现', '同意', '都要', '搜索', '一致', '线上', '球队', '养老', '声音', '摄影', '各地', '自我', '也要', '扩大', '石油', '净值', '千万', '特别是', '离开', '东方', '职位', '下一', '有利', '变成', '贫困', '大小', '透露', '担心', '专门', '绿色', '直播', '想到', '上方', '反映', '主席', '完整', '准确', '奖励', '震荡', '安徽', '承担', '清楚', '当天', '整理', '引发', '本报告', '自行', '重视', '广大', '生物', '警方', '唯一', '讨论', '人物', '思维', '不够', '电脑', '本人', '经销', '万人', '地铁', '及其', '或许', '也许', '梦想', '美联', '技巧', '自主', '反弹', '更好', '它们', '身上', '引导', '时也', '称为', '预测', '利于', '语言', '专项', '恢复', '公路', '对手', '财富', '原文', '阳光', '现车', '止损', '放弃', 'ff', 'App', 'IM', 'Phon', 'LED', 'iPh', 'IC', 'SV', 'ap', 'ple', 'ak', 'na', 'GB', 'TF', 'ub', 'OPE', 'TI', 'uc', 'VS', 'eW', 'BC', 'tch', 'CR', 'up', 'app', 'iz', 'ir', 'EN', 'CI', 'GL', 'con', 'gg', 'SC', 'XX', 'MC', 'per', 'ON', 'ook', 'by', 'eg', 'wit', 'lu', 'rac', 'VSU', 'ong', 'eA', 'IN', 'ES', 'NO', 'cc', 'eT', 'hen', 'ght', 'US', 'RO', 'PV', 'KK', 'CEO', 'rp', 'eP', 'SA', 'CB', 'du', 'rk', 'ind', 'TO', 'MB', 'SD', 'ny', 'II', 'eM', 'DI', 'BCD', 'CV', 'ik', 'IS', 'SI', 'rc', 'mag', 'rC', 'qu', 'nf', 'Wat', 'ABCD', 'ws', 'ge', 'ACD', 'BM', 'NG', 'SL', 'DJ', 'rf', 'Note', 'RM', 'min', 'DS', 'eC', 'nk', 'PPA', 'FF', 'AD', 'cm', 'SM', 'ER', 'un', 'su', 'BI', 'BS', 'KD', 'mb', 'eB', 'MP', 'CN', 'FI', 'ix', 'gu', 'ME', 'CF', 'MI', 'ug', 'ru', 'eG', 'ib', 'xx', 'ECO', 'LO', 'dth', 'Pro', 'tra', 'ND', 'OF', 'uan', 'au', 'cti', 'And', 'OR', 'VI', 'tS', 'AL', 'DR', 'OC', 'nS', 'fu', 'man', 'IR', 'ack', 'sw', 'PM', 'din', 'DC', 'DK', '09', '0201', '28', '102', '99', '55', '82', '35', '002', '29', '26', '92', '88', '36', '56', '500', '53', '98', '27', '48', '212', '44', '58', '43', '66', '412', '202', '38', '49', '300', '65', '68', '小米', '论坛', '粉丝', '葡萄', '租赁', '航天', '志愿', '药品', '少年', '第一次', '全新', '出口', '财产', '二十', '本公司', '维持', '通常', '革命', '两年', '负债', '一句', '三年', '看出', '配合', '冠军', '的价格', '余额', '年底', '互动', '权益', '公益', '房价', '每日', '反应', '会员', '板块', '生在', '新华', '基金份额', '重组', '重的', '能是', '电池', '然是', '至今', '手术', '例如', '治理', '扶贫', '逐步', '授权', '带着', '多种', '红色', '个性', '众多', '百度', '独特', '定了', '复制', '控股', '在此', '广场', '才是', '青年', '实际上', 'pnb', 'TM', 'mc', 'TE', 'ber', 'Fi', 'iF', 'ul', 'Wi', 'son', 'kin', 'SP', 'CP', 'OM', 'oy', 'ard', 'ud', 'pm', 'SB', 'af', 'art', 'tw', 'ment', 'um', 'CA', 'ess', 'ax', 'CL', 'Mar', 'bu', 'lc', 'dy', 'ETF', 'ho', 'AG', 'CS', 'CM', 'DA', 'EIA', 'ps', 'ged', 'EE', 'ine', 'ls', 'san', 'sy', '78', '54', '67', '59', '列车', '期货', '三星', '印度', '发表', '调节', '涨幅', '停车', '改造', '观点', '地位', '运用', '公交', '复杂', '的设计', '无人', '农民', '朋友圈', '认真', '出版', '实践', '赛季', '名字', '个股', '留学', '主体', '留言', '一代', '组成', '依据', '寻找', '日报', '人工', '承诺', 'SG', 'di', 'km', 'PT', 'td', 'tf', 'SO', 'rm', 'SR', 'oth', 'LG', 'OOOO', 'CH', 'nn', 'ie', 'Uni', 'ca', 'NBA', 'IV', 'ms', 'IF', 'db', 'lus', 'ire', 'Win', 'OT', 'ot', '醫院', '铁路', '合肥', '简介', '运输', '出去', '传播', '成熟', '青岛', '检察', '厦门', '武汉', '产品的', '成果', '出台', '工艺', '收取', '眼睛', '死亡', '增速', '高中', '还会', '数据显示', '国外', '下去', '出席', '订阅', '一体', '表决', '思考', '劳动', '很有', '加速', '原来', '手段', '领先', '商场', '识别关注', 'BR', 'EM', 'DN', 'PU', 'new', 'boo', 'VC', 'VIP', 'IPO', 'DO', 'API', 'MG', 'PD', 'tc', 'gy']
    print(len(voc_set))
    print(len(pair_priority))
    for k, v in voc_set.items():
        final_voc_dict[k] = id
        id+=1
    for pair in pair_priority:
        if pair in final_voc_dict: print(f'abnormal: {pair}')
        final_voc_dict[pair] = id
        id+=1
    print(f'Combined length: {len(final_voc_dict)}|{len(voc_set)}|{len(pair_priority)}')
    print(final_voc_dict)


def trans_t2id(one_str, token_4sets, unk_percent=0.05):
    id_list = []
    idx = 0
    max_len = 4  # 原逻辑的最大匹配长度
    str_len = len(one_str)
    unk_limit_count = str_len * unk_percent
    unk_count = 0

    while idx < str_len:
        matched = False
        # 计算当前可匹配的最大长度（避免越界）
        current_max_possible_len = min(max_len, str_len - idx)
        # 从最长到最短匹配token
        for current_len in range(current_max_possible_len, 0, -1):
            sub_str = one_str[idx:idx + current_len]
            if sub_str in token_4sets[current_len]:
                id_list.append(token2idx[sub_str])
                idx += current_len
                matched = True
                break
        # 未匹配则添加UNK_ID，指针后移1位
        if not matched:
            id_list.append(UNK_ID)
            idx += 1
            unk_count+=1
            if unk_count >= unk_limit_count:
                return []

    return id_list


def get_voc_4_set():
    token_sets = {1: set(), 2: set(), 3: set(), 4: set()}
    for token, idx in token2idx.items():
        token_len = len(token)
        if 1 <= token_len <= 4: token_sets[token_len].add(token)
    return token_sets

def process_batch_lines(lines, token_4sets, filter_length, blank_pattern):
    """批量处理多行数据，返回id_chunk和计数"""
    batch_id_chunk = []
    batch_process = 0
    batch_skip = 0
    #logging.info(f'process_batch_lines working...')
    for line_bytes in lines:
        # 原逐行逻辑（JSON解析、正则、过滤、转id）
        line = line_bytes.decode('utf-8').strip()
        if not line:
            batch_skip +=1
            continue
        try:
            data = json.loads(line)
        except:
            batch_skip +=1
            continue

        content = blank_pattern.sub('', data['content'])
        if len(content) <= filter_length:
            batch_skip +=1
            continue
        str_list = trans_t2id(content, token_4sets)
        if str_list:
            batch_id_chunk.extend(str_list)
            batch_id_chunk.append(EOS_ID)
            batch_process +=1
        else:
            batch_skip +=1
    del lines
    gc.collect()
    return batch_id_chunk, batch_process, batch_skip

def split_file2chunk(file_path, filter_length=64, batch_size=10000000, max_chunk_length = 100000000, max_workers=8):
    try:
        # 初始化多进程池（默认用CPU核心数）
        max_workers = max_workers or os.cpu_count()
        pool = Pool(processes=max_workers)
        results = []  # 存储多进程返回的结果

        id_chunk = []
        idx = 0
        process_line_count = 0
        skip_line_count = 0

        token_4sets = get_voc_4_set()
        blank_pattern = re.compile(r'\s+')
        with open(file_path, "rb") as file:
            while True:
                lines = file.readlines(batch_size)
                if not lines: break
                # 提交批量任务到进程池
                #logging.info(f'submit task: {len(lines)}')
                res = pool.apply_async(
                    process_batch_lines,
                    args=(lines, token_4sets, filter_length, blank_pattern)
                )
                results.append(res)

                if len(results) >= max_workers:
                    res = results.pop(0)
                    #logging.info(f'deal one task')
                    batch_id, batch_p, batch_s = res.get()  # 阻塞获取结果
                    #logging.info(f'res: {len(batch_id)}|{batch_p}|{batch_s}')
                    id_chunk.extend(batch_id)
                    process_line_count += batch_p
                    skip_line_count += batch_s

                    if len(id_chunk) >= max_chunk_length:
                        chunk_path = f'./cache_data/news_ids_chunk_{idx}.pkl'
                        logging.info(f'Chunk-{idx}: {len(id_chunk)}, process|skip: {process_line_count}|{skip_line_count}')
                        save_tmp_data(chunk_path, id_chunk)
                        idx += 1
                        id_chunk = []
                        process_line_count=0
                        skip_line_count=0


        for res in results:
            batch_id, batch_p, batch_s = res.get()  # 阻塞获取结果
            id_chunk.extend(batch_id)
            process_line_count += batch_p
            skip_line_count += batch_s
            if len(id_chunk) >= max_chunk_length:
                chunk_path = f'./cache_data/news_ids_chunk_{idx}.pkl'
                logging.info(f'Chunk-{idx}: {len(id_chunk)}, process|skip: {process_line_count}|{skip_line_count}')
                save_tmp_data(chunk_path, id_chunk)
                idx += 1
                id_chunk = []
                process_line_count = 0
                skip_line_count = 0

        if id_chunk:
            chunk_path = f'./cache_data/news_ids_chunk_{idx}.pkl'
            logging.info(f'[Last]Chunk-{idx}: {len(id_chunk)}, process|skip: {process_line_count}|{skip_line_count}')
            save_tmp_data(chunk_path, id_chunk)

        pool.close()
        pool.join()

    except Exception as e:
        print(f"split_file2chunk Error：{e}")
    return

def tmp_verify():
    src_file = f'./cache_data/sft_qa_ids_src.pkl'
    tgt_file = f'./cache_data/sft_qa_ids_tgt.pkl'
    src_list = load_tmp_data(src_file)
    tgt_list = load_tmp_data(tgt_file)
    print(f'data length: {len(src_list)}|{len(tgt_list)}')

    for i in range(20):
        print(src_list[i])
        print(tgt_list[i])

def tmp_verify2():
    d_list = load_tmp_data('./cache_data/news_array_int16_chunk_0.pkl')
    or_size1 = sys.getsizeof(d_list)
    d_l1 = array.array('q', d_list)
    d_l2 = array.array('i', d_list)
    d_l3 = d_list.tolist()
    dl_size1 = sys.getsizeof(d_l1)
    dl_size2 = sys.getsizeof(d_l2)
    dl_size3 = sys.getsizeof(d_l3)
    print(f'or_size1: {or_size1}, size1: {dl_size1}, size2: {dl_size2}, size3: {dl_size3}')

    print(d_list[0:1000])
    str_list=[]
    for i, t in enumerate(d_list):
        if t == UNK_ID:
            for tmp_t in d_list[i-10:i+10]:
                str_list.append(idx2token[tmp_t])
            print(''.join(str_list))
            str_list = []
        if i >= 10000: break

def trans_file_l2a():
    for i in range(22):
        in_file = f'./cache_data/news_ids_chunk_{i}.pkl'
        out_file = f'./cache_data/news_array_int16_chunk_{i}.pkl'
        d_list = load_tmp_data(in_file)
        d_list = array.array('h', d_list)
        save_tmp_data(out_file, d_list)

def pre_dal_qa_file():
    file_path = './local_datasets/baike_qa2019/baike_qa_train.json'
    abnormal_line = 0
    short_line = 0
    long_line = 0
    blank_pattern = re.compile(r'\s+')
    token_4sets = get_voc_4_set()
    sft_src = []
    sft_tgt = []
    i = 0
    with open(file_path, "rb") as file:
        while True:
            line = file.readline()
            if not line: break

            line_str = line.decode('utf-8').strip()
            if not line_str:
                abnormal_line += 1
                continue

            try:
                data = json.loads(line)
            except:
                abnormal_line += 1
                continue

            title = blank_pattern.sub('', data['title'])
            desc = blank_pattern.sub('', data['desc'])
            answer = blank_pattern.sub('', data['answer'])

            question = desc if desc else title
            if not answer:
                #print(f'no answer q: {question}')
                abnormal_line += 1
                continue

            # 问题过长，占用一半以上，舍弃
            if len(question) > BLOCK_SIZE//2:
                #print(f'question too long: {question}')
                abnormal_line += 1
                continue
            else:
                q_id = trans_t2id(question, token_4sets)

            if len(answer) > BLOCK_SIZE:
                a_id = trans_t2id(answer[0:BLOCK_SIZE], token_4sets)
            else:
                a_id = trans_t2id(answer, token_4sets)

            if not q_id or not a_id:
                abnormal_line += 1
                continue

            # 可以直接达成错位
            src_id = q_id + [SEP_ID] + a_id
            tgt_id= [IGN_LOSS_ID]*len(q_id) + a_id + [EOS_ID]

            #qa_len = len(q_id) + len(a_id)
            #assert(len(src_id) == len(tgt_id)), "unexpected length"

            if len(src_id) > BLOCK_SIZE:
                src_id = src_id[0:BLOCK_SIZE]
                tgt_id = tgt_id[0:BLOCK_SIZE-1]+[EOS_ID]
                long_line += 1
                #print(f'Long Q_{long_line}: {question}')
                #print(f'Long Q_ID_{long_line}: {q_id}')
                #print(f'Long A_{long_line}: {answer}')
                #print(f'Long A_ID_{long_line}: {a_id}')
            else:
                left_num = BLOCK_SIZE - len(src_id)
                src_id = src_id + [PAD_ID] * left_num
                tgt_id = tgt_id + [IGN_LOSS_ID] * left_num
                short_line += 1
                #print(f'Short Q_{short_line}: {question}')
                #print(f'Short Q_ID_{short_line}: {q_id}')
                #print(f'Short A_{short_line}: {answer}')
                #print(f'Short A_ID_{short_line}: {a_id}')

            # 转array（int 16），减少保存和内存
            src_id_array = array.array('h', src_id)
            tgt_id_array = array.array('h', tgt_id)
            sft_src.append(src_id_array)
            sft_tgt.append(tgt_id_array)
            #print(f'src_id_array: {len(src_id_array)}|{src_id_array}')
            #print(f'tgt_id_array: {len(tgt_id_array)}|{tgt_id_array}')
            #i += 1
            #if i > 10: break

    print(f'Res: {abnormal_line}|{short_line}|{long_line}')
    src_file = f'./cache_data/sft_qa_ids_src.pkl'
    tgt_file = f'./cache_data/sft_qa_ids_tgt.pkl'
    save_tmp_data(src_file, sft_src)
    save_tmp_data(tgt_file, sft_tgt)


def pre_dal_qa_file2():
    file_path = './local_datasets/baike_qa2019/baike_qa_train.json'
    abnormal_line = 0
    blank_pattern = re.compile(r'\s+')
    category_dict = dict()
    good_qa_pair=[]
    with open(file_path, "rb") as file:
        while True:
            line = file.readline()
            if not line: break

            line_str = line.decode('utf-8').strip()
            if not line_str:
                abnormal_line += 1
                continue

            try:
                data = json.loads(line)
            except:
                abnormal_line += 1
                continue

            category = blank_pattern.sub('', data['category'])
            title = blank_pattern.sub('', data['title'])
            desc = blank_pattern.sub('', data['desc'])
            answer = blank_pattern.sub('', data['answer'])

            question = desc if desc else title
            if not answer:
                # print(f'no answer q: {question}')
                abnormal_line += 1
                continue

            # 找问题长度适中的, 128 取 32(Q) + 96(A)
            if (3 < len(question) <= 32) and (3 < len(answer) <= 96):
                good_qa_pair.append((question, answer))
                category_cut = category.split('-')
                first_tag=category_cut[0]
                second_tag = 'NA'
                if len(category_cut) > 1: second_tag = category_cut[1]
                third_tag = 'NA'
                if len(category_cut) > 2: third_tag = category_cut[2]
                if len(category_cut) > 3: print(f'abnormal category {category}')

                if first_tag in category_dict:
                    if second_tag in category_dict[first_tag]:
                        category_dict[first_tag][second_tag][third_tag] = category_dict[first_tag][second_tag].get(third_tag, 0) + 1
                    else:
                        category_dict[first_tag][second_tag] = {third_tag: 1}
                else:
                    category_dict[first_tag] = {second_tag:{third_tag: 1}}
            else:
                abnormal_line += 1

    print(f'Good: {len(good_qa_pair)}|{abnormal_line}')
    print(f'{category_dict}')


def pre_dal_qa_file3(filter_word):
    file_path = './local_datasets/baike_qa2019/baike_qa_train.json'
    abnormal_line = 0
    blank_pattern = re.compile(r'\s+')
    token_4sets = get_voc_4_set()
    sft_src = []
    sft_tgt = []
    good_qa_pair = []
    with open(file_path, "rb") as file:
        while True:
            line = file.readline()
            if not line: break

            line_str = line.decode('utf-8').strip()
            if not line_str:
                abnormal_line += 1
                continue

            try:
                data = json.loads(line)
            except:
                abnormal_line += 1
                continue

            category = blank_pattern.sub('', data['category'])
            if filter_word not in category:
                abnormal_line += 1
                continue
            title = blank_pattern.sub('', data['title'])
            desc = blank_pattern.sub('', data['desc'])
            answer = blank_pattern.sub('', data['answer'])

            question = desc if desc else title
            if not answer:
                #print(f'no answer q: {question}')
                abnormal_line += 1
                continue

            # 问题过长，占用一半以上，舍弃
            if not ((3 < len(question) < 32*1.5) and (3 < len(answer) < 96*1.5)):
                abnormal_line += 1
                continue
            else:
                q_id = trans_t2id(question, token_4sets)
                a_id = trans_t2id(answer, token_4sets)

            if not q_id or not a_id:
                abnormal_line += 1
                continue

            # 可以直接达成错位
            src_id = q_id + [SEP_ID] + a_id
            tgt_id= [IGN_LOSS_ID]*len(q_id) + a_id + [EOS_ID]

            if len(src_id) > BLOCK_SIZE:
                print(f'Over 128 seq: {question}:{answer}')
                abnormal_line += 1
                continue
            else:
                left_num = BLOCK_SIZE - len(src_id)
                src_id = src_id + [PAD_ID] * left_num
                tgt_id = tgt_id + [IGN_LOSS_ID] * left_num

            # 转array（int 16），减少保存和内存
            src_id_array = array.array('h', src_id)
            tgt_id_array = array.array('h', tgt_id)
            sft_src.append(src_id_array)
            sft_tgt.append(tgt_id_array)
            good_qa_pair.append((question, answer))


    print(f'Res: {abnormal_line}|{len(good_qa_pair)}|{len(sft_src)}|{len(sft_tgt)}')
    src_file = f'./cache_data/sft_pc_qa_ids_src.pkl'
    tgt_file = f'./cache_data/sft_pc_qa_ids_tgt.pkl'
    txt_file = f'./cache_data/sft_pc_qa.txt'
    save_tmp_data(src_file, sft_src)
    save_tmp_data(tgt_file, sft_tgt)
    save_txt_data(txt_file, good_qa_pair)



if __name__ == '__main__':
    #filter_by_threshold(2850)
    #dataloader = load_tmp_data('./local_datasets/data_loader.pkl')
    #voc_set = load_tmp_data('./cache_data/ok_dict_55.pkl')
    #add_vocab()
    #combine_voc()
    #file_path = './local_datasets/new2016zh/news2016zh_train.json'
    #split_file2chunk(file_path)
    #trans_file_l2a()
    #tmp_verify()
    #pre_dal_qa_file()
    #pre_dal_qa_file2()
    pre_dal_qa_file3('电脑/网络')