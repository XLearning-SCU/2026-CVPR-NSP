import datetime
import argparse

if __name__ == '__main__':
    # 接收命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, required=True)
    parser.add_argument('--seed', type=str, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--ng', type=int, required=True)
    args = parser.parse_args()

    # 获取当前时间
    now = datetime.datetime.now()
    date = now.strftime("%m%d-%H%M%S")

    # 拼接由 时间 + TDS + IDS 组成的文件夹名
    # 例如: 0520-103000_tds3_ids5
    name = f"{date}-b{args.bs}-sd{args.seed}-{args.lr:.1e}-ng{args.ng}-"

    print(name)