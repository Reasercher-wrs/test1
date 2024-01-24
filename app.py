import gradio as gr
import theme
import requests
import secrets
import string
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import BitsAndBytesConfig

# pip install gradio==3.50.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
os.system("pip install gradio==3.50.2 -i https://pypi.tuna.tsinghua.edu.cn/simple")
#os.system("pip install bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple")

default_css = """\
<style type="text/css">
    .diff {
        border: 1px solid #cccccc;
        background: none repeat scroll 0 0 #f8f8f8;
        font-family: 'Bitstream Vera Sans Mono','Courier',monospace;
        font-size: 12px;
        line-height: 1.4;
        white-space: normal;
        word-wrap: break-word;
    }
    .diff div:hover {
        background-color:#ffc;
    }
    .diff .control {
        background-color: #eaf2f5;
        color: #999999;
    }
    .diff .insert {
        background-color: #ddffdd;
        color: #000000;
    }
    .diff .insert .highlight {
        background-color: #aaffaa;
        color: #000000;
    }
    .diff .delete {
        background-color: #ffdddd;
        color: #000000;
    }
    .diff .delete .highlight {
        background-color: #ffaaaa;
        color: #000000;
    }
</style>
"""

title = "<h1 style='text-align: center; color: #333333; font-size: 40px;'> 🔎 LLM-Detector </h1>"
description = """
LLM-Detector 是一个AI/LLMs生成的文本检测器。我们支持中/英文内容输入进行检测，目前支持文档级内容检测。
"""

style = theme.Style()

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

def infer(select_model, input_text):
    predictions = []
    
    folder_path = './data/'
    file_name = generate_random_string(32)
    
    # LLM-Detector ZH服务
    if select_model=="LLM-Detector-Small-zh":
        tokenizer_zh = AutoTokenizer.from_pretrained("./LLM-Detector-Small-zh", trust_remote_code=True)
        # int4
        #model_zh = AutoModelForCausalLM.from_pretrained("./ZH", device_map="auto", trust_remote_code=True, quantization_config = BitsAndBytesConfig(load_in_4bit=True)).cuda()
        # GPU
        #model_zh = AutoModelForCausalLM.from_pretrained("./ZH", device_map="auto", trust_remote_code=True).eval()
        model_zh = AutoModelForCausalLM.from_pretrained("./LLM-Detector-Small-zh", device_map="auto", trust_remote_code=True).cuda()
        # CPU
        #model_zh = AutoModelForCausalLM.from_pretrained("./ZH", device_map="cpu", trust_remote_code=True).eval()
        
        llm_results, history = model_zh.chat(tokenizer_zh, input_text, history=None)

        if llm_results=="Human":
            predicted_labels = "👨‍🔧 这段文本由人类生成！"
            predictions.append((str(input_text), "人类"))
        if llm_results=="AI":
            predicted_labels = "🤖 这段文本由AI生成！"
            predictions.append((str(input_text), "AI"))
        
        json_data = {
                "text": str(input_text),
                "label": llm_results
        }
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        with open(folder_path+file_name+'.json', 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)
        gr.Textbox.update(visible=False)
        return predicted_labels, predictions, folder_path+file_name+'.json'
    
    if select_model=="LLM-Detector-Small-en":
        tokenizer_en = AutoTokenizer.from_pretrained("./LLM-Detector-Small-en", trust_remote_code=True)
        # GPU
        #model_en = AutoModelForCausalLM.from_pretrained("./EN", device_map="auto", trust_remote_code=True).eval()
        model_en = AutoModelForCausalLM.from_pretrained("./LLM-Detector-Small-en", device_map="auto", trust_remote_code=True).cuda()
        # CPU
        #model_en = AutoModelForCausalLM.from_pretrained("./EN", device_map="cpu", trust_remote_code=True).eval()
        
        llm_results, history = model_en.chat(tokenizer_en, input_text, history=None)
        
        if llm_results=="Human":
            predicted_labels = "👨‍🔧 这段文本由人类生成！"
            predictions.append((str(input_text), "人类"))
        if llm_results=="AI":
            predicted_labels = "🤖 这段文本由AI生成！"
            predictions.append((str(input_text), "AI"))
        
        json_data = {
                "text": str(input_text),
                "label": llm_results
        }
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        with open(folder_path+file_name+'.json', 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)
    
        return predicted_labels, predictions, folder_path+file_name+'.json'
    else:
        return "请选择模型", "请选择模型", folder_path+file_name+'.json'

with gr.Blocks(theme=style) as demo:
    #gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            select_model = gr.Dropdown(["LLM-Detector-Small-zh", "LLM-Detector-Small-en"], label="模型选择", info="请选择适合文本内容的检测模型(🍽️目前支持中文/英文检测，标记模型即将推出)：", value='LLM-Detector-Small-zh')
            input_text = gr.Textbox(label="文本内容", lines=20, autofocus=True, info="请输入你的文本内容，LLM-Detector将判断它是由AI还是人类生成的：", show_copy_button=True, placeholder="请输入您的文本...")
            submit = gr.Button("开始检测", variant="primary")
            gr.Markdown("- 🔴继续则表示您**同意并接受我们收集您提交的文本内容**！请注意，我们对于**通过检测获得的结果不负责任**。\n - 🔵该项目**仅用于学术研究**不可用于商业用途！")
            
        with gr.Column(): # "Results", open=False
            label = gr.Label(label="预测结果")
            highlighted_prediction = gr.HighlightedText(
                                    label="文本分析",
                                    combine_adjacent=True,
                                    color_map={"AI": "red", "人类": "green"},
                                    show_legend=True)
            save_data = gr.File(label="保存结果")

    submit.click(infer, inputs=[select_model, input_text], outputs=[label, highlighted_prediction, save_data])
    
    gr.Markdown("## 👁️‍🗨️示例测试")
    gr.Markdown("#### 人类文本示例")
    gr.Examples(
        [
        ["LLM-Detector-Small-zh", "目前,基于机器视觉的车辆检测算法存在检测速度较慢的问题。针对该问题,提出一种基于YOLO算法的车辆实时检测方法。YOLO检测算法的基本模型由卷积层,池化层以及全连接层组成,具有强鲁棒性以及能够快速完成车辆检测任务。选择交通监控视频作为数据集进行车辆检测试验。结果表明,YOLO检测算法的查准率为89.3%,查全率为81.0%,检测速度达到60f/s,基本满足交通监控中车辆检测的实时性需求,说明该方法合理可行。运用该方法与2种不同的检测算法进行对比分析,得出YOLO算法的检测速度最快。"],
        ["LLM-Detector-Small-zh", "12月5日，国家主席习近平致电安德里·尼里纳·拉乔利纳，祝贺他当选连任马达加斯加共和国总统。习近平指出，中国同马达加斯加传统友好。近年来，在我们共同引领下，两国关系加速发展，各领域交流合作成果丰硕，双方在涉及彼此核心利益和重大关切问题上坚定相互支持。我高度重视中马关系发展，愿同拉乔利纳总统一道努力，继续推动中马全面合作伙伴关系取得更大发展，更好造福两国人民。"],
        ["LLM-Detector-Small-zh","谢谢指点迷津，两个time改为365，小元延续时间能长一点，我准备改为3600试试。经过思考，rm E2E目录后重新建立运行文件，time改为365，克制自己不问复杂问题，chatbot能坚持下来。说句实在话，不算反应速度，这个水平比chatgpt2.0稍差，比开始的豆包和360智脑能稍好一点（这话只限于昇腾论坛讨论，只是个人看法），考虑到开发板的价格，说明软件优化很好，证明华为硬件开加chatbot表现超乎想象。我买了板子开始试验仅仅一周时间，试验结果说明昇腾组件确实非同凡响。期待能开放一些调整参数，或者开放训练库的组成和训练方法，我们可以做一些学习辅助，过去说好记性不如烂笔头，现在有了chatbot胜过好记性，希望昇腾组件能推广开来，有更广泛的应用。"],
        ["LLM-Detector-Small-zh", "我们知道，当事件A的发生对事件B的发生有影响时，条件概率P(B|A)和概率 P(B)一般是不相等的，但有时事件A的发生看上去对事件B的发生没有影响，比如依次抛掷两枚硬币，抛掷第一枚硬币的结果(事件A)应该对第二枚硬币的结果(事件B)没有影响，这时P(B|A)与P(B)相等吗?"],
        ["LLM-Detector-Small-zh", "由超几何分布和二项分布等离散型随机变量的分布，我们知道离散型随机变量的分布列能够完全描述随机变量取值的概率规律。但是，在许多实际问题中，还需要了解离散型随机变量的某种特征，例如离散型随机变量的平均取值大小和取值的集中程度。我们把这种反映概率分布的某种特征的数值，叫做离散型随机变量的数字特征，下而我们来介绍两个最基本的离散型随机变量的数字特征."],
        ["LLM-Detector-Small-en","In Vauban, Germany, citizens have made the decision to not use cars. To some people, this may be something that they could never imagine themselves doing, because it would make life more complicated. However, these people are pleased with their decision and would not have it any other way. Furthermore, there are many advantages to making this change. By limiting car usage, citizens can improve their own health and economic state.\n\nBy making the decision to stop using cars, one can become healthier, both mentally and physically. One citizen who has already taken this step said,\"when I had a car I was always tense. I'm much happier this way\" Rosenthal 3. Many people who chose to limit their car usage, decided to walk or ride bikes instead. By chosing the alternative, they are less stressed. There is something soothing about walking down the road in a quiet and peaceful environment. Walking gives one time to reflect and think, while driving requires concentration and can be stressful. In addition to improving one's mental health, limiting car usage can also improve one's physical health. Pollution from the air can take a toll on someone's physical health and the environment around them. \"Passenger cars are responsible for 12 percent of greenhouse gas emissions in Europe... and up to 50 percent in some carintensive areas in the United States\" Rosenthal 5. Pollution poisons the air of most cities where people live and breathing the pollution is not healthy for an individual. In cities like Beijing, inhabitants wear air filters over their mouths in hopes fo escaping the pollution. Limiting car usage can help improve air quality quickly. For example, after having multiple days of intense smog, Paris decided to ban cars with evennumbered plates for one day. After this one day of limited car usage, \"the smog cleared enough Monday for the ruling French party to rescind the ban for oddnumbered plates on Tuesday\" Duffer 19.\n\nAnother advantage to limiting cars is that it could mean economic improvement for individuals and countries. The banning of cars can mean improvement in the appearance of cities, which can have positive impacts on the economy of cities. \"Parks and sports centers also have bloomed throughout the city uneven, pitted sidewalks have beeen replaced by broad, smooth sidewalks\" Selsky 28. These improvements in the city can draw more people to them and stimulate the economy in places that have had difficulty before. In addition, individuals can save money by carpooling, biking, walking, or using public transit as an alternative optopn to driving. During the 2013 rececession, people were forced to sell their cars due to lack of money. However, after they recovered from this, they decided not to return to car usage due to their content in the lifestyle they had chosen Rosenthal 32 In conclusion, limiting one's usage of cars can have only positive impacts on one's life. This decision can have lasting impacts one's happiness, the environment, and the economy."],
        ["LLM-Detector-Small-en","Dear state senator, I'm writing to you today regarding my concerns on our voting method for the president of the United States. Although we've been voting by the electoral college for how ever many years, I don't think it is the most efficient and fair way of voting. Our Chamber of Commerce, former vice president Richard Nixon and many more would have to agree with me when I say that abolishing the electoral college could only be beneficial to us. The electoral college system is unfair, confusing and forces people to compromise.\n\nThe electoral college is unfair, being that voters don't always control who their electors vote for, opposed to election by popular vote. One reason why America strives is the fact that we are a democracy, where every one gets a say and we are not ruled by a dictators or communist. The electoral college in no way follows our democratic system, the people are not voting for our president our electors are the ones voting for us.\n\nNot only is the electoral college unfair but it is also confusing. For new voters they may be confused by the electoral college. New voters may wonder why can't I just vote for the candidate I most prefer. Think about it like this, in the electoral college the electors are the middle man. Why not cut the middle man out? And as a result make the voting system much simpler.\n\nPeople may agrue that the electoral college system stops a majority vote. So let's say, you're a democrat living in the state of Texas with the electoral college system in place. You might as well not vote for an elector cause the majority of the people in texas are going to vote for the republican elector. On the other hand, there is the election by popular vote, gives everyone a say in whom they'd like to vote for. There is always the possibility of the disater factor.\n\nAfter sharing my concerns with you state senator, I hope you understand where I am coming from."],
        ],
        
        inputs=[select_model, input_text]
    )
    
    gr.Markdown("#### GPT-4文本示例")
    gr.Examples(
        [
        ["LLM-Detector-Small-zh", "正态分布，也常被称为高斯分布，是统计学中最为重要的概率分布之一。想象一下，我们在班级里测量所有同学的身高。你可能会发现，大部分同学的身高会集中在一个平均值附近，而很高或很矮的同学则相对较少。这种现象，其中大多数数据围绕一个中心值聚集，并且向两边逐渐减少，形成一个钟形曲线，就是正态分布的典型特征。"],
        ["LLM-Detector-Small-zh","Python是一种高级编程语言，以其易于学习和使用的语法、强大的库和框架支持，以及广泛的应用领域（如网站开发、数据科学、人工智能和自动化）而闻名。"]
        ],
        inputs=[select_model, input_text]
    )
    
    gr.Markdown("#### ChatGPT文本示例")
    gr.Examples(
        [
        ["LLM-Detector-Small-zh", "在小镇上，一只丢失的小猫引发了一场惊心动魄的寻找。孩子们齐心协力，通过海报和社交媒体传播消息。最终，一位善良的邻居找到了小猫，小镇因此洋溢着欢笑和感激的气氛。"],
        ["LLM-Detector-Small-zh", "机器学习是一种人工智能（AI）的分支，它致力于通过从数据中学习模式和规律，让计算机系统具备自主学习的能力，而无需明确地进行编程。通过训练模型，机器学习使计算机能够从经验中学到并提高性能，以执行特定任务，如分类、预测、识别模式等。在机器学习中，算法通过不断优化自身以适应新数据，从而改善其性能。"],
        ["LLM-Detector-Small-en","I strongly believe that the Electoral College should remain the way it is or, better yet, that we should elect the president by popular vote. This is due to the fact that the Electoral College does not accurately reflect the will of the people. For example, in the 2016 presidential election, an estimated two million more people voted for Hillary Clinton than for Donald Trump however, Trump won the Electoral College vote, 304 to 232. This means that a candidate can win a majority of the Electoral College voters while losing the popular vote! Furthermore, voting for President should be an individual citizen decision, not a state decision. The Electoral College works by awarding all of a state's electoral votes to the winner of the majority of votes in the state. This means that a candidate can win the majority of votes in a state and still not receive any of that states electoral votes. This goes against the concept of onepersononevote, since a candidate can win the majority of votes in a state and still not win any electoral votes. By eliminating the Electoral College and electing the president by popular vote, the votes of every individual will be counted, and the candidate who wins the most votes nationally will win the election. In conclusion, the Electoral College does not reflect the will of the people and votes in state are not equally weighted. It is time to elect the president by popular vote and to finally give the votes of individual citizens the weight they deserve."],
        ],
        inputs=[select_model, input_text]
    )
    
    gr.Markdown("#### 文心一言文本示例")
    gr.Examples(
        [
        ["LLM-Detector-Small-zh", "大语言模型（LLM）是指使用大量文本数据训练的深度学习模型，可以生成自然语言文本或理解语言文本的含义。大语言模型可以处理多种自然语言任务，如文本分类、问答、对话等，是通向人工智能的一条重要途径。大语言模型的特点是规模庞大，包含数十亿的参数，帮助它们学习语言数据中的复杂模式。这些模型通常基于深度学习架构，如转化器，这有助于它们在各种NLP任务上取得令人印象深刻的表现。目前的大语言模型（如GPT和BERT）采用与小模型类似的Transformer架构和预训练目标（如Language Modeling），与小模型的主要区别在于增加模型大小、训练数据和计算资源。"],
        ["LLM-Detector-Small-zh", "尊敬的经理，我写这封邮件是想向您申请在接下来的三天内请假。由于家庭原因，我需要处理一些私事，因此希望能够得到您的理解和支持。我保证在请假期间尽一切努力确保我的工作不会影响到团队的正常运作。我已经做好了工作安排，并与同事协商好了相关工作事宜。如果您能够批准我的请假申请，我将不胜感激。如果您有任何疑问或需要进一步的信息，请随时与我联系。谢谢您的理解和支持。顺祝商祺！"],
        ["LLM-Detector-Small-en","My name is XX, a passionate and outgoing individual. I have a strong interest in learning new things and am always eager to explore new experiences and opportunities. With a Bachelor's degree in Computer Science, I possess a strong foundation in technology and enjoy utilizing my skills to solve complex problems. As a self-motivated and goal-oriented person, I thrive in fast-paced and challenging environments. I take pride in my adaptability and ability to quickly grasp new concepts, regardless of the domain. With my strong communication skills, I can easily work with colleagues and clients to achieve common goals. In my free time, I enjoy reading books, playing basketball, and traveling to new places. I believe that travel is an excellent way to gain perspective and broaden one's horizons. In conclusion, I am a skilled professional with a strong educational background and a broad range of skills. I am confident in my abilities and am always looking for opportunities to grow and develop professionally. "]
        ],
        inputs=[select_model, input_text]
    )
    
    gr.Markdown("## 🥨特殊测试")
    gr.Markdown("#### 标点符号")
    gr.Examples(
        [
        ["LLM-Detector-Small-zh", "你好，请问有什么需要帮助"],
        ["LLM-Detector-Small-zh", "你好，请问有什么需要帮助？"],
        ],
        inputs=[select_model, input_text]
    )
    
    gr.HTML("<img src=\"https://img.lecter.one/i/2023/12/08/irw9rv.png\">")
    gr.HTML("<center><a href=\'https://clustrmaps.com/site/1bxsq\'  title=\'Visit tracker\'><img src=\'//clustrmaps.com/map_v2.png?cl=4148bc&w=a&t=tt&d=_GWhYr_Z4pFr-TUk17Zm_vc3VtJsdHrhC9WCDT5wNFo&co=ffffff&ct=441616\'/></a></center>")
    gr.HTML("<center><p>© 2023 LLM-Detector. If you have any issue, feel free to contact us by <a href=\"mailto:wrs6@88.com\">wrs6@88.com</a>. Our Github is <a href=\"https://github.com/QiYuan-tech\">QiYuan-tech</a>.</p></center>")
    
theme=gr.themes.Base()
demo.title = "LLM-Detector 🚀"
demo.queue(max_size=6)         
demo.launch(debug=True, show_api=False, server_port=80, server_name="0.0.0.0")
