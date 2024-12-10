import bisect
from itertools import combinations
import parameters as pam
Mc = pam.Mc
holes = pam.holes
Ni_position = pam.Ni_position
O1_orbs = pam.O1_orbs
O2_orbs = pam.O2_orbs
O_orbs = pam.O_orbs
Ni_orbs = pam.Ni_orbs


def filter_less_d7_states(All_states):
    """
    过滤掉d7以下的态
    :param All_states: All_states = ((x1, y1, orb1, s1), (x2, y2, orb2, s2), ...), 表示多个空穴的态
    :return: logic(逻辑值, True表示保留, False表示过滤)
    """
    logic = True
    at_Ni_num = {key: 0 for key in Ni_position}
    for state in All_states:
        position = state[:2]
        if position in at_Ni_num:
            at_Ni_num[position] += 1
    for position, num in at_Ni_num.items():
        if num > 3:
            logic = False
            break
    return logic


def exist_Sz_states(All_states):
    """
    保留在Sz_set(具体看parameters.py文件)的态, Sz_set = [...]
    :param All_states:All_states = ((x1, y1, orb1, s1), (x2, y2, orb2, s2), ...), 表示多个空穴的态
    :return:True表示总自旋满足Sz_set, False表示不满足
    """
    if 'All_states' in pam.Sz_set:
        return True
    Sz_value = map(lambda state: 1 / 2 if state[-1] == 'up' else -1 / 2, All_states)
    Sz_total = sum(Sz_value)
    if Sz_total in pam.Sz_set:
        return True
    else:
        return False


def combination_filter(All_states):
    """
    过滤d7以下的态并保留总自旋满足Sz_set的态
    :param All_states:All_states = ((x1, y1, orb1, s1), (x2, y2, orb2, s2), ...), 表示多个空穴的态
    :return:逻辑值
    """
    return filter_less_d7_states(All_states) and exist_Sz_states(All_states)


def create_lookup_tbl():
    """
    创建lookup_tbl，用于查找态所对应的索引
    :return: O_states: 只有一个空穴在O上的态, Ni_states: 只有一个空穴在Ni上的态
    states_one: 一个空穴的态, lookup_tbl: 排序过后的态列表, 包含多个空穴的所有态
    """
    O_states = []
    Ni_states = []
    # 先创建只有一个空穴在O上的态, 并将O分为两种类型, O1和O2, 两种类型的跳跃轨道不同
    for y in range(-Mc, Mc + 1):
        x_max = Mc - abs(y) + 1
        for x in range(-x_max, x_max + 1, 2):
            # O1的y坐标是偶数，O2的y坐标是奇数
            if y % 2 == 0:
                for orb in O1_orbs:
                    for s in ['up', 'dn']:
                        O_states.append((x, y, orb, s))
            else:
                for orb in O2_orbs:
                    for s in ['up', 'dn']:
                        O_states.append((x, y, orb, s))
    # 再创建只有一个空穴在Ni上的态
    for (x3, y3) in Ni_position:
        for orb3 in Ni_orbs:
            for s3 in ['up', 'dn']:
                Ni_states.append((x3, y3, orb3, s3))
    # 合并O_states和Ni_states, 得到一个空穴的态, 并排序
    states_one = sorted(O_states + Ni_states)
    # holes个空穴的所有态, 组合方式按照states_one的排序
    lookup_tbl = combinations(states_one, holes)
    # 过滤掉d7以下的态并保留总自旋满足Sz_set的态
    lookup_tbl = filter(combination_filter, lookup_tbl)
    lookup_tbl = list(lookup_tbl)
    return O_states, Ni_states, states_one, lookup_tbl


def check_in_vs_condition(x, y):
    """
    判断state是否处在VS空间中
    :param x: 空穴的横坐标
    :param y: 空穴的纵坐标
    :return: 逻辑值
    """
    distance_Ni1 = abs(x + 1) + abs(y)
    distance_Ni2 = abs(x - 1) + abs(y)
    if distance_Ni1 <= Mc or distance_Ni2 <= Mc:
        return True
    else:
        return False


def count_inversions(states):
    """
    计算元组中元素的逆序数(即经过多少次可以将该元组变为升序)
    (只适合第一个元素不是按照升序排列的情况)
    :param states: states = ((x1, y1, orb1, s1), (x2, y2, orb2, s2), ...), 表示多个空穴的态
    :return: inversions
    """
    inversions = 0
    for value in states[1:]:
        if states[0] < value:
            break
        inversions += 1
    return inversions


def count_inversions_for_disorder(states):
    """
    计算元组中元素的逆序数(即经过多少次可以将该元组变为升序)
    (适合乱序排列的一般情况)
    :param states: states = ((x1, y1, orb1, s1), (x2, y2, orb2, s2), ...), 表示多个空穴的态
    :return: inversions
    """
    inversions = 0
    for i in range(1, holes):
        behind_state = states[i]
        for front_state in states[: i]:
            if front_state > behind_state:
                inversions += 1
    return inversions


class VariationalSpace:
    """
    定义变分空间类
    """
    def __init__(self):
        self.Mc = Mc
        self.O_states, self.Ni_states, self.states_1, self.lookup_tbl = create_lookup_tbl()
        self.dim = len(self.lookup_tbl)
        print('Variational space dimension:  ', self.dim)

    def get_index(self, state):
        """
        根据state, 在lookup_tle中查找对应lookup_tle的索引
        :param state: ((x1, y1, orb1, s1), (x2, y2, orb2, s2), (x3, y3, orb3, s3)...)
        :return: index, 如果没有找到，则返回None
        """
        dim = self.dim
        lookup_tbl = self.lookup_tbl
        index = bisect.bisect_left(lookup_tbl, state)
        # 可能存在比lookup_tbl中所有元素都大的state, 所以需要判断index是否等于dim, 防止超出索引范围
        if index == dim:
            return None
        elif lookup_tbl[index] == state:
            return index
        else:
            return None
