"""Deterministic long-document probes; neither contains external application data."""
NEEDLE = '73291584'


def repeated_document(records):
    filler = '这是一条无关的背景记录，请继续阅读后面的记录。\n'
    return (filler * (records // 2) + f'校验编号为{NEEDLE}。' + filler * (records - records // 2)
            + '\n只回答校验编号的八位数字。')


def inventory_document(records):
    rows = [f'第{i}条库存记录：仓库{chr(65+i%20)}在第{i%28+1}日登记了{i%83+2}件样品。'
            f'样品类别为{["金属构件", "木质托盘", "玻璃瓶", "纸箱", "布袋"][i%5]}，'
            f'接收区位于{["东侧", "西侧", "南侧", "北侧"][i%4]}。检验完成后，将对应记录归档，等待下一批货物。\n'
            for i in range(records)]
    return ('以下是参考资料。\n' + ''.join(rows[:records//2]) + f'\n唯一的验收校验编号是{NEEDLE}。\n'
            + ''.join(rows[records//2:]) + '\n请从参考资料中提取唯一的验收校验编号，只输出八位数字。')


DOCUMENTS = {'repeated': repeated_document, 'varied_inventory': inventory_document}
