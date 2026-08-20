import pandas as pd
import re
from pathlib import Path
from Levenshtein import distance as levenshtein_distance
import Levenshtein
from cn2an import cn2an, an2cn
import json


# ==================== 函数 3：文本清洗函数 ====================
def clean_text(text: str) -> str:
    """通用文本清洗函数"""
    if pd.isna(text) or not isinstance(text, str):
        return ''
    # 1. 去除无效标签
    text = text.replace('【地址】', '').replace('【机器人】', '')
    if 'unk' in text:
        return ''
    # 2. 移除【xxx】格式的标签
    pattern_label = r'【.*?】'
    if re.search(pattern_label, text):
        return ''
    # 3. 保留中英文、数字，去除其他符号
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    text = text.lower()
    # 4. 口语词替换（按最长匹配优先）
    replace_map = {
        '奥': '哦', '噢': '哦', '喔': '哦', '昂': '哦',
        '哦哦': '哦',
        '呃': '哎', '诶': '哎', '唉': '哎', '喂': '哎', '哎哎': '哎',
        '阿': '啊', '呀': '啊', '吖': '啊', '呐': '啊', '啊啊': '啊',
        '恩': '嗯', '摁': '嗯', ' 蒽': '嗯', '嗯嗯': '嗯',
        '嘛': '吗', '勒': '嘞'
    }
    # replace_map = {
    #     '奥': '嗯', '噢': '嗯', '喔': '嗯', '昂': '嗯',
    #     '哦哦': '嗯',
    #     '呃': '嗯', '诶': '嗯', '唉': '嗯', '喂': '嗯', '哎哎': '嗯',
    #     '阿': '嗯', '吖': '嗯', '呐': '嗯', '啊啊': '嗯',
    #     '恩': '嗯', '摁': '嗯', '蒽': '嗯', '嗯嗯': '嗯',
    #     '嘛': '吗', '勒': '嘞','哦':'嗯',"额":'嗯',"额额":'嗯',
    #     '哎': "嗯"
    # }
    slang_words = [
        '哦哦', '哦', '噢', '喔', '昂',
        '呃', '诶', '唉', '喂', '哎哎', '哎',
        '阿', '呀', '吖', '呐', '啊啊', '啊',
        '恩', '摁', '蒽', '嗯嗯', '嗯',
        '嘛', '勒'
    ]
    pattern = '|'.join(re.escape(key) for key in sorted(replace_map.keys(), key=len, reverse=True))
    # pattern = '|'.join(re.escape(word) for word in sorted(slang_words, key=len, reverse=True))
    # text = re.sub(pattern, lambda m: replace_map[m.group(0)], text)
    # text = re.sub(pattern, '', text)
    # 5. 数字转换：阿拉伯数字 -> 汉字
    text = convert_numbers_to_chinese(text)
    return text


# ==================== 函数 1：生成预测文本文件 ====================
def generate_prediction_file(input_path: str, id_col: str, pred_col: str, output_path: str = "prediction_temp.xlsx"):
    """从文件读取预测文本，保留原始与清洗后文本"""
    file_path = Path(input_path)
    if not file_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    if input_path.endswith(".xlsx") or input_path.endswith(".xls"):
        df = pd.read_excel(input_path, sheet_name='Sheet1')
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".txt"):
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip().split(' ') for line in f if line.strip()]
        for l in lines:
            if len(l)>2:
                print(l)
        df = pd.DataFrame(lines, columns=[id_col, pred_col])
    elif input_path.endswith(".json"):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
    else:
        raise ValueError("不支持的文件格式，仅支持: .xlsx, .csv, .txt, .json")

    if id_col not in df.columns:
        raise ValueError(f"列 {id_col} 不存在于输入文件中")
    if pred_col not in df.columns:
        raise ValueError(f"列 {pred_col} 不存在于输入文件中")

    output_df = pd.DataFrame({
        'id': df[id_col],
        'predict_text_raw': df[pred_col].astype(str),
        'predict_text': df[pred_col].astype(str).apply(clean_text)
    })
    output_df.to_excel(output_path, index=False)
    print(f"✅ 预测文本已生成，保存至: {output_path}")
    return output_path


# ==================== 函数 2：生成标注文本文件 ====================
def generate_label_file(input_path: str, id_col: str, label_col: str, output_path: str = "label_temp.xlsx"):
    """从文件读取标注文本，保留原始与清洗后文本"""
    file_path = Path(input_path)
    if not file_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
   
    if input_path.endswith(".xlsx") or input_path.endswith(".xls"):
        df = pd.read_excel(input_path, sheet_name='Sheet1')
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".txt"):
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip().split(' ') for line in f if line.strip()]
        df = pd.DataFrame(lines, columns=[id_col, label_col])
    elif input_path.endswith(".json"):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
    else:
        raise ValueError("不支持的文件格式，仅支持: .xlsx, .csv, .txt, .json")

    if id_col not in df.columns:
        raise ValueError(f"列 {id_col} 不存在于输入文件中")
    if label_col not in df.columns:
        raise ValueError(f"列 {label_col} 不存在于输入文件中")

    output_df = pd.DataFrame({
        'id': df[id_col],
        'label_text_raw': df[label_col].astype(str),
        'label_text': df[label_col].astype(str).apply(clean_text)
    })
    output_df.to_excel(output_path, index=False)
    print(f"✅ 标注文本已生成，保存至: {output_path}")
    return output_path


# ==================== 函数 3：中文数字转汉字（辅助） ====================
def convert_numbers_to_chinese(text: str) -> str:
    """将阿拉伯数字转为中文数字（如 "123" → "一百二十三"）"""
    result = ""
    i = 0
    while i < len(text):
        char = text[i]
        if char.isdigit():
            num_str = ""
            while i < len(text) and text[i].isdigit():
                num_str += text[i]
                i += 1
            try:
                chinese_num = an2cn(num_str)
                result += chinese_num
            except Exception as e:
                result += num_str
            continue
        else:
            result += char
        i += 1
    return result


# ==================== 主函数：串联整个流程 ====================
def main(
    pred_input: str, pred_id_col: str, pred_text_col: str,
    label_input: str, label_id_col: str, label_text_col: str,
    output_path: str = "result_word_acc.xlsx"
):
    """主函数：读取预测与标注文件 → 清洗 → 计算字准率 → 输出结果"""
    print("🚀 开始处理字准率计算...")

    # 1. 生成预测文件
    pred_file = generate_prediction_file(pred_input, pred_id_col, pred_text_col, "prediction_temp.xlsx")
    pred_df = pd.read_excel(pred_file)

    # 2. 生成标注文件
    label_file = generate_label_file(label_input, label_id_col, label_text_col, "label_temp.xlsx")
    label_df = pd.read_excel(label_file)

    # 3. 合并预测与标注（基于 id）
    merged = pd.merge(pred_df, label_df, on='id', how='outer', suffixes=('_pred', '_label'))

    # 4. 计算每条样本的 dis, len, word_acc
    def calc_metrics(row):
        

        pred_clean = row['predict_text']
        label_clean = row['label_text']
        if pd.isna(pred_clean):
            pred_clean = ""  # 确保是字符串
        if pd.isna(label_clean):
            return pd.Series({'dis': len(pred_clean), 'len': 0, 'word_acc': 0,'S': 0,'D': len(pred_clean),'I': 0})
        ops = Levenshtein.editops(pred_clean, label_clean)
        s = sum(1 for op in ops if op[0] == 'replace')
        d = sum(1 for op in ops if op[0] == 'delete')
        i = sum(1 for op in ops if op[0] == 'insert')
        dis = levenshtein_distance(pred_clean, label_clean)
        label_len = len(label_clean)
        # dis_clipped = min(dis, label_len)
        word_acc = 1 - (dis / label_len) if label_len > 0 else 1.0
        return pd.Series({'dis': dis, 'len': label_len, 'word_acc': word_acc,'S': s,'D': d,'I': i})

    
    metrics = merged.apply(calc_metrics, axis=1)
    merged = pd.concat([merged, metrics], axis=1)

    # 5. 计算总体字准率：1 - (sum(dis) / sum(len))
    total_dis = merged['dis'].sum()
    total_len = merged['len'].sum()
    total_S = merged['S'].sum()
    total_D = merged['D'].sum()
    total_I = merged['I'].sum()
    overall_word_acc = 1 - (total_dis / total_len) if total_len > 0 else 1.0

    # ✅ 新增：只在结果末尾添加一个统计行（不重复每行）
    stats_row = pd.DataFrame([{
        'id': '总体统计',
        'predict_text_raw': '',
        'predict_text': '',
        'label_text_raw': '',
        'label_text': '',
        'dis': total_dis,
        'len': total_len,
        'S': total_S,
        'D': total_D,
        'I': total_I,
        'word_acc': overall_word_acc,
        'overall_word_acc': overall_word_acc  # 可选：保留该列用于展示
    }])

    # 合并结果 + 统计行
    final_df = pd.concat([merged, stats_row], ignore_index=True)

    # 6. 输出结果
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        final_df.to_excel(writer, index=False, sheet_name='结果')

        # 可选：在另一个 sheet 中存放统计摘要
        summary_df = pd.DataFrame({
            '指标': ['总样本数', '总编辑距离', '总标签长度', '总体字准率'],
            '数值': [
                len(merged),
                total_dis,
                total_len,
                f"{overall_word_acc:.4f}"
            ]
        })
        summary_df.to_excel(writer, index=False, sheet_name='统计摘要')

    print(f"✅ 所有计算完成，结果已保存至: {output_path}")

    # 7. 打印统计信息
    print("\n📊 统计信息:")
    print(f"总样本数: {len(merged)}")
    print(f"平均字准率: {merged['word_acc'].mean():.4f}")
    print(f"平均编辑距离: {merged['dis'].mean():.2f}")
    print(f"平均标签长度: {merged['len'].mean():.2f}")
    print(f"总体字准率 (1 - sum(dis)/sum(len)): {overall_word_acc:.4f}")



# ==================== 使用示例 ====================

# pred_input="字准率/frcrn-out.csv",
if __name__ == "__main__":
    main(
        # pred_input="字准率/dns64-out.csv",
        # pred_input="字准率/frcrn-out.csv",
        # pred_input="字准率/base-out.csv",
        # pred_input="字准率/rnnoise-out.csv",
        # pred_input="字准率/nr-out.csv",
        # pred_input="字准率/zip-out.csv",
        # pred_input="字准率/nr85-out.csv",
        # pred_input="字准率/nr-frcrn-out.csv",
        # pred_input="字准率/distillmos-out.csv",
        # pred_input="字准率/科技badcase.csv",
        # pred_input="字准率/科技goodcase.csv",
        # pred_input="字准率/自研goodcase.csv",
        pred_input="字准率/自研badcase.csv",
        pred_id_col="id",
        pred_text_col="text",
        # label_input="/nfsc/ph_iss_id118703_vol1001_dev/xijujun/标注数据/fenshen降噪/fenshen审批混合音频.xlsx",
        # label_input="fenshen/label.xlsx",
        # label_input="fenshen/label_good.xlsx",
        label_input="fenshen/label_bad.xlsx",
        label_id_col="ID",
        label_text_col="asr正确文本",
        # label_text_col="IT标注文本",
        
        # output_path="字准率/base字准率.xlsx"
        # output_path="字准率/dns64字准率.xlsx"
        # output_path="字准率/frcrn字准率.xlsx"
        # output_path="字准率/rnnoise字准率.xlsx"
        # output_path="字准率/nr字准率.xlsx"
        # output_path="字准率/自研goodcase字准率.xlsx"
        output_path="字准率/自研badcase字准率.xlsx"
    )