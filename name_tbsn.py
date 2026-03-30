import datetime
import argparse

if __name__ == '__main__':
    # 接收命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, required=True)
    parser.add_argument('--ps', type=int, required=True)
    parser.add_argument('--seed', type=str, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--chosen', type=int, required=True)
    args = parser.parse_args()

    # 获取当前时间
    now = datetime.datetime.now()
    date = now.strftime("%m%d-%H%M%S")

    # 拼接由 时间 + TDS + IDS 组成的文件夹名
    # 例如: 0520-103000_tds3_ids5
    name = f"{date}-sd{args.seed}-{args.lr:.1e}-b{args.bs}p{args.ps}-ch{args.chosen}-"

    print(name)