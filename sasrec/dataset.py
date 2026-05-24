import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        # 激活的mm_emb特征ID，"激活"这个词说白了就是你选择用哪几种特征。你有6种多模态特征（81-86），但不一定每次训练都全用：
        #后续主函数会写个这个，默认值是[81,82]，也就是默认只用两种多模态特征。可以根据需要选择用更多或更少的特征，但要确保数据文件中有对应的特征数据。
        #parser.add_argument('--mm_emb_id', nargs='+', default=['81','82'], type=str, choices=[str(s) for s in range(81, 87)])
        self.mm_emb_ids = args.mm_emb_id

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        ##待优化
        ##normalize参数控制是否进行z-score标准化，默认True。标准化可以加速模型训练收敛，但会改变特征的分布，可能对某些模型性能有影响。可以根据实际情况选择是否标准化。
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids, normalize=True)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        #反转 indexer['i'] 字典中键值对的映射关系。
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """+
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        #这句话是根据用户ID在数据文件中找到对应的行，
        #然后读取该行的数据。
        # 因为数据文件是按行存储的，每行对应一个用户的数据，
        # 所以通过文件偏移量seq_offsets这里存着，可以快速定位到用户数据所在的位置，避免了从头读取整个文件的开销。
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        # 负采样核心逻辑：生成一个不在当前序列s中且存在于item_feat_dict中的随机物品ID
        t = np.random.randint(l, r)
        # 循环条件：1) t在序列s中（避免选中正样本）；2) t不在item_feat_dict中（确保物品有特征）
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，Padding 就是“补齐”。生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        # 函数主要步骤：
        # 1. 加载用户原始序列数据
        # 2. 重构序列：将user和item作为不同token类型混合
        # 3. 初始化输出数组（左padding准备）
        # 4. 收集序列中所有物品ID（用于负采样）
        # 5. 左padding循环：从后往前填充序列，生成正负样本
        # 6. 处理缺失特征，返回9个数组供模型训练使用

        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        #将 user 和 item 作为不同的 token 类型混合在同一个序列中，并且为每个 token 生成对应的特征。这样模型在处理序列时就能同时利用用户和物品的信息。
        # 这样就构建了一个混合了用户和物品信息的序列，模型在训练时可以同时学习用户和物品的特征表示。
        ext_user_sequence = []
        for record_tuple in user_sequence:
            #使用 _ 是一种约定俗成的方式，表示这个变量不会在后续代码中使用，是那个时间戳字段，我们在这个函数里不需要它，所以用 _ 来占位。
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                #代码逻辑：如果用户ID和用户特征存在，就把它作为一个token插入到序列的前面（左边），并标记token_type为2，actiontype保留；
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                # 如果物品ID和物品特征存在，就把它作为一个token插入到序列的后面（右边），并标记token_type为1。
                ext_user_sequence.append((i, item_feat, 1, action_type))

        #下面这些全都是一个格式化的过程，就是把上面构建的混合序列填充成固定长度maxlen+1，创建一个形状为 [self.maxlen + 1] 的全零数组，并且生成对应的pos、neg样本和特征。这个过程比较复杂，主要是为了适应模型输入的格式要求。
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                #tuple的2号位置，是第三个，那就是type位置，如果是1的话就i说明是item，如果这个item_id不为0，就把它加入ts集合。
                # 这个ts集合就是用来记录用户序列中出现过的item_id的，后面负采样的时候要保证采样的负样本不在这个集合里。
                ts.add(record_tuple[0])

        nxt = ext_user_sequence[-1]  # 初始化"下一个"元素为序列最后一个元素
        idx = self.maxlen  # 从数组末尾开始填充，实现左padding（不足maxlen的部分在前面补0）

        # left-padding核心逻辑：从后往前遍历序列，将用户序列填充到maxlen+1的长度
        # 为什么要从后往前？因为左padding需要在序列前面补0，从后往前填充可以自然实现这一点
        for record_tuple in reversed(ext_user_sequence[:-1]):
            #截取这个列表，但是“不要”最后一个元素也就是从倒数第二个开始呗，因为数组切片是左闭右开的
            # 这个循环的目的是为了构建模型输入的序列数据，进行left-padding。
            # 对于每个token，根据它是用户还是物品，填充对应的ID、特征、token类型等信息。
            # 同时根据下一个token的信息生成正样本和负样本，以及它们的特征。
            #倒二
            i, feat, type_, act_type = record_tuple
            #倒一
            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            # 正负样本生成：只有当下一个token是物品(type=1)且不是padding位置(next_i≠0)时才需要
            # 为什么只对物品生成样本？因为序列推荐的核心是预测下一个物品，用户token不需要预测
            if next_type == 1 and next_i != 0:
                # 正样本：下一个真实访问的物品
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                # 负采样：随机选择一个不在当前序列中的物品作为负样本
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                # 获取负样本的特征（从item_feat_dict中获取并填充缺失值）
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        # 将None特征替换为默认值（处理padding位置的特征）
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        # 返回9个数组，对应模型训练的不同需求：
        # 1) seq: 当前序列token ID       2) pos: 正样本ID         3) neg: 负样本ID
        # 4) token_type: 当前token类型   5) next_token_type: 下一个token类型  6) next_action_type: 下一个动作类型
        # 7) seq_feat: 当前token特征     8) pos_feat: 正样本特征   9) neg_feat: 负样本特征
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
            其实应该是:num_samples: 数据集样本量 (int)
            这个和上面之前init定义的usernum不一定相等，因为一个用户可能会有多条序列数据，或者有些用户数据可能被过滤掉了。
            这个是用offsets求长度算出来的，也就是说有几条交互数据，我现在的len就是多少，而不是用户数量。
            seq_offsets.pkl中存储了每条交互数据在原始数据文件中的偏移量，因此len(self.seq_offsets)实际上是交互数据的数量，而不是用户的数量。
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        # 用户稀疏特征：分类特征，用整数表示，默认值填充为0
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        # 物品稀疏特征：分类特征，用整数表示，默认值填充为0
        feat_types['item_sparse'] = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        # 物品数组特征：多值特征，用列表表示，默认值填充为[0]
        feat_types['item_array'] = []
        # 用户数组特征：多值特征，用列表表示，默认值填充为[0]
        feat_types['user_array'] = ['106', '107', '108', '110']
        # 物品多模态嵌入特征：预训练的多模态嵌入向量，默认值填充为全零向量
        feat_types['item_emb'] = self.mm_emb_ids
        # 用户连续特征：连续数值特征，默认值填充为0
        feat_types['user_continual'] = []
        # 物品连续特征：连续数值特征，默认值填充为0
        feat_types['item_continual'] = []

        # 为每种特征类型设置默认值和统计信息（特征数量）
        # 这些默认值会在fill_missing_feat函数中使用
        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        # 多模态嵌入特征特殊处理：使用全零向量作为默认值，维度与嵌入向量相同
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        注意：这个函数不显式区分用户和物品特征，而是通过特征类型系统隐式区分：
        1. 特征类型在 _init_feat_info() 中已明确分类为 user_sparse, item_sparse, user_array, item_array, item_emb 等
        2. 统一处理逻辑：所有特征缺失都用默认值填充，只有多模态嵌入（item_emb）需要特殊处理
        3. 通过 item_id 参数控制多模态嵌入处理：对于用户特征，item_id 是用户ID，不会进入 item_emb 处理分支
        4. 这种设计避免了代码重复，保持了逻辑简洁性

        Args:
            feat: 特征字典，可能包含部分特征ID的键值对
            item_id: 物品ID（对于用户特征，这个参数实际上是用户ID，但名称保持为item_id）

        Returns:
            filled_feat: 填充后的完整特征字典，包含所有预定义特征ID
        """
        # 1. 初始化特征字典：如果输入特征为None，则初始化为空字典
        if feat == None:
            feat = {}
        filled_feat = {}
        # 浅拷贝：保留原始特征字典中的所有现有特征
        for k in feat.keys():
            filled_feat[k] = feat[k]

        # 2. 收集所有预定义的特征ID并计算缺失字段
        all_feat_ids = []
        # 从feature_types中提取所有特征ID（包括user和item的各种类型特征）
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        # 计算缺失的特征字段：预定义特征ID集合 - 当前特征字典中已有的键集合
        missing_fields = set(all_feat_ids) - set(feat.keys())
        # 3. 为缺失的特征填充默认值（统一处理用户和物品特征）
        for feat_id in missing_fields:
            # 根据特征类型使用对应的默认值：稀疏特征=0，数组特征=[0]，多模态嵌入=零向量等
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        # 4. 特殊处理多模态嵌入特征（只属于物品特征，类型为item_emb）
        for feat_id in self.feature_types['item_emb']:
            # 条件检查：1) item_id不为0（排除padding位置）；2) 物品在多模态嵌入字典中存在
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                # 确保嵌入值是numpy数组类型，然后用实际嵌入向量替换零向量默认值
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]
                    # 注意：对于用户特征，item_id实际上是用户ID，但不会进入这个分支，
                    # 因为用户特征不属于item_emb类型，feature_types['item_emb']中不包含用户特征ID

        # 5. 返回填充后的完整特征字典
        return filled_feat

    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
        """
        # 将batch数据解压：zip(*batch)将多个样本的对应字段分组
        # batch是一个列表，每个元素是__getitem__返回的9个数组的元组
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        # 将数值型数组转换为torch.Tensor：先通过np.array()将元组转换为numpy数组，再转换为torch.Tensor
        # 这些张量的形状为 [batch_size, maxlen+1]
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))

        # 特征数据保持为列表：因为每个元素是字典，后续在模型中会进一步处理
        # 这些列表的长度为batch_size，每个元素是形状为[maxlen+1]的对象数组，对象是特征字典
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_ = record_tuple
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
        """
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_feat = list(seq_feat)

        return seq, token_type, seq_feat, user_id


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids, normalize=True):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表
        normalize: 是否进行z-score标准化

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}

    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        loaded_count = 0
        error_count = 0

        # 用于标准化的统计量
        all_values = [] if normalize else None

        # 1. 首先尝试从目录读取（适用于所有特征）
        base_path = Path(mm_path, f'emb_{feat_id}_{shape}')

        # 处理85特征的不一致：代码中是3584，目录名可能是4096
        if not base_path.exists() and feat_id == '85':
            # 尝试查找实际的目录名
            possible_paths = list(mm_path.glob(f'emb_{feat_id}_*'))
            if possible_paths:
                base_path = possible_paths[0]
                print(f"注意: 特征 #{feat_id} 使用实际目录 {base_path.name}，而非代码中的维度 {shape}")

        if base_path.exists() and base_path.is_dir():
            # 查找part-*文件（无扩展名）或*.json文件
            part_files = list(base_path.glob('part-*'))
            if not part_files:
                part_files = list(base_path.glob('*.json'))

            if not part_files:
                print(f"警告: 特征 #{feat_id} 目录 {base_path} 中没有找到part-*或*.json文件")
            else:
                for part_file in sorted(part_files):
                    try:
                        with open(part_file, 'r', encoding='utf-8') as file:
                            line_num = 0
                            for line in file:
                                line_num += 1
                                line = line.strip()
                                if not line:  # 跳过空行
                                    continue

                                try:
                                    data = json.loads(line)
                                except json.JSONDecodeError as e:
                                    print(f"JSON解析错误 {part_file}:{line_num}: {e}")
                                    error_count += 1
                                    continue

                                # 检查必需的键
                                if 'anonymous_cid' not in data:
                                    print(f"缺少键 'anonymous_cid' {part_file}:{line_num}")
                                    error_count += 1
                                    continue

                                if 'emb' not in data:
                                    print(f"缺少键 'emb' {part_file}:{line_num}")
                                    error_count += 1
                                    continue

                                anonymous_cid = data['anonymous_cid']
                                emb_array = data['emb']

                                # 确保emb是numpy数组
                                if isinstance(emb_array, list):
                                    emb_array = np.array(emb_array, dtype=np.float32)
                                elif not isinstance(emb_array, np.ndarray):
                                    print(f"emb格式错误 {part_file}:{line_num}: 期望list或ndarray，得到{type(emb_array)}")
                                    error_count += 1
                                    continue

                                emb_dict[anonymous_cid] = emb_array
                                if normalize and all_values is not None:
                                    all_values.append(emb_array)
                                loaded_count += 1

                    except Exception as e:
                        print(f"读取文件 {part_file} 错误: {e}")
                        error_count += 1

        # 2. 如果目录方式失败，尝试pickle文件（向后兼容，特别是对于81特征）
        if not emb_dict and feat_id == '81':
            pkl_path = Path(mm_path, f'emb_{feat_id}_{shape}.pkl')
            if pkl_path.exists():
                try:
                    with open(pkl_path, 'rb') as f:
                        emb_dict = pickle.load(f)
                    print(f"✅ Loaded #{feat_id} mm_emb from pickle: {len(emb_dict)} items")
                    # 收集pickle文件中的值用于标准化
                    if normalize and all_values is not None:
                        for emb_array in emb_dict.values():
                            if isinstance(emb_array, np.ndarray):
                                all_values.append(emb_array)
                            elif isinstance(emb_array, list):
                                all_values.append(np.array(emb_array, dtype=np.float32))
                except Exception as e:
                    print(f"读取pickle文件 {pkl_path} 错误: {e}")
                    error_count += 1

        # 3. 检查是否成功加载
        if emb_dict:
            # 如果需要标准化，对特征进行z-score标准化
            if normalize and all_values and len(all_values) > 0:
                try:
                    # 计算均值和标准差
                    all_arrays = np.vstack(all_values)
                    mean = np.mean(all_arrays, axis=0)
                    std = np.std(all_arrays, axis=0)
                    # 避免除以0，将std小于1e-8的位置设为1
                    std[std < 1e-8] = 1.0

                    # 标准化emb_dict中的所有值
                    standardized_count = 0
                    for key, value in emb_dict.items():
                        if isinstance(value, np.ndarray):
                            emb_dict[key] = (value - mean) / std
                            standardized_count += 1
                        elif isinstance(value, list):
                            emb_dict[key] = ((np.array(value, dtype=np.float32) - mean) / std).tolist()
                            standardized_count += 1

                    print(f'  标准化特征 #{feat_id}: {standardized_count}个向量 (均值={mean.mean():.4f}, 标准差={std.mean():.4f})')
                except Exception as e:
                    print(f'  标准化特征 #{feat_id} 时出错: {e}')

            mm_emb_dict[feat_id] = emb_dict
            print(f'✅ Loaded #{feat_id} mm_emb: {loaded_count} items' +
                  (f' (跳过 {error_count} 个错误行)' if error_count > 0 else ''))
        else:
            print(f'❌ 无法加载特征 #{feat_id}，请检查数据文件')
            mm_emb_dict[feat_id] = {}

    return mm_emb_dict
