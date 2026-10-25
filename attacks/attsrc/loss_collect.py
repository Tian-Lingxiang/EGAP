import matplotlib.pyplot as plt


class LossCollector:
    def __init__(self):
        # 使用字典来存储每个 key 对应的损失值列表
        self.losses = {}

    def add(self, key: str, value: "float|list|dict"):
        """插入一组值, 如果key不存在则创建一个新的列表, 可以是 float、list 或 dict"""
        if isinstance(value, list):
            # 如果 value 是 list 类型，递归插入每个元素
            for i, item in enumerate(value):
                self.add(f"{key}.{i}", item)
        elif isinstance(value, dict):
            # 如果 value 是 dict 类型，递归插入每个键值对
            for subkey, subvalue in value.items():
                self.add(f"{key}.{subkey}", subvalue)
        elif isinstance(value, (float, int)):
            key = str(key)
            if key not in self.losses:
                self.losses[key] = []
            self.losses[key].append(value)
        else:
            raise TypeError(
                f"value={value} must be int, float, list or dict. not {type(value)}"
            )

    def plot(
        self,
        path: str,
        log: bool = True,
        scatter: bool = False,
        collect_every: int = 1,
        select: "None|str|list" = None,
    ):
        """在同一张图上绘制指定的折线，保存为图像
        * log: y轴是否为对数坐标
        * scatter: 是否为散点图 (或折线图)
        * collect_every: 每批大小
        * select: 控制绘制哪些折线：
            - None: 绘制所有折线
            - str: 绘制指定的 key 对应的折线
            - list: 绘制多个指定 key 对应的折线
        """
        plt.figure(figsize=(10, 6))

        # 如果 select 是 None 或者是 list，则绘制所有的折线
        if select is None:
            keys_to_plot = self.losses.keys()
        elif isinstance(select, str):
            keys_to_plot = [select]  # 只绘制一个指定的折线
        else:
            keys_to_plot = select  # 绘制指定的多个折线

        # 遍历选择的 key 和对应的损失值
        for key in keys_to_plot:
            if key in self.losses:
                values = self.losses[key]
                x_vals = [
                    i * collect_every for i in range(1, len(values) + 1)
                ]  # 横坐标根据 collect_every 缩放
                if log:
                    plt.yscale("log")  # 设置 y 轴为对数坐标
                if scatter:
                    plt.scatter(x_vals, values, label=key)
                else:
                    plt.plot(x_vals, values, label=key)

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plt.legend()
        plt.grid(True)

        # 保存图像
        try:
            plt.savefig(path)
        except OSError:
            print(f"Cannot open '{path}'")
        plt.close()
