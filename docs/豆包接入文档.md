本接口可将音频文件异步转写成文本，最大支持 512 MB且时长不超过 5 小时的音频文件，支持raw、wav、mp3、ogg 、pcm 、spx、amr、aac、m4a格式

客户端上传音频文件后，服务端异步处理转写任务，识别结果需通过配套的[查询接口](https://docs.volcengine.com/docs/6561/2606792?lang=zh)获取

&nbsp;

<span data-label="purple">POST</span>https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit

&nbsp;


<span id="ULAS3Id9"></span>
### 请求头


**X\-Api\-Key ** `string` <span data-api-tag="require|1RbFw3">必选</span>

API Key 可从 [控制台>API Key管理](https://console.volcengine.com/speech/new/setting/apikeys?projectName=default.) 获取

<div data-tips="true" data-tips-type="default" data-tips-is-title="true">说明</div>


<div data-tips="true" data-tips-type="default">同时支持<a href="https://console.volcengine.com/speech/service/10035">旧版控制台</a>的鉴权方式，详见<a href="https://www.volcengine.com/docs/6561/2534847?lang=zh">旧版控制台鉴权参考示例</a></div>




**X\-Api\-Resource\-Id ** `string` <span data-api-tag="require|W2vM70">必选</span>

请求的模型版本，可选值：


* `volc.seedasr.auc`：豆包录音文件识别模型2.0

* `volc.bigasr.auc`：豆包录音文件识别模型1.0



**X\-Api\-Request\-Id ** `string` <span data-api-tag="require|W2vM70">必选</span>

提交和查询任务的任务ID，推荐传入随机生成的UUID



**X\-Api\-Sequence ** `string` <span data-api-tag="require|W2vM70">必选</span>

发包序号，固定值: `-1`




<span id="OG7QrhRG"></span>
### 请求体


**audio ** `dict` <span data-api-tag="require|WnVo1B">必选</span>


**url ** `string` <span data-api-tag="require|1zS7c2">必选</span>

指定音频链接



**language ** `string`

指定识别语种，当前支持识别以下语种：


* 中文普通话：`zh-CN`

* 英语：`en-US`

* 日语：`ja-JP`

* 印尼语：`id-ID`

* 西班牙语：`es-MX`

* 葡萄牙语：`pt-BR`

* 德语：`de-DE`

* 法语：`fr-FR`

* 韩语：`ko-KR`

* 菲律宾语：`fil-PH`

* 马来语：`ms-MY`

* 泰语：`th-TH`

* 阿拉伯语：`ar-SA`

* 意大利语：`it-IT`

* 孟加拉语：`bn-BD`

* 希腊语：`el-GR`

* 荷兰语：`nl-NL`

* 俄语：`ru-RU`

* 土耳其语：`tr-TR`

* 越南语：`vi-VN`

* 波兰语：`pl-PL`

* 罗马尼亚语：`ro-R0`

* 尼泊尔语：`ne-NP`

* 乌克兰语：`uk-UA`

* 粤语：`yue-CN`


<div data-tips="true" data-tips-type="default" data-tips-is-title="true">说明</div>


<div data-tips="true" data-tips-type="default">当 <code>language</code> 参数为空时，模型支持识别以下语种：中文、英文、上海话、闽南话、四川话、陕西话、粤语</div>




**format ** `string` <span data-api-tag="require|7sHXWE">必选</span>

指定音频格式

可选值： `wav` / `mp3` / `ogg `/ `pcm` /` spx` / `amr` /` aac` / `m4a`



**codec** `string`

指定音频编码格式，默认为raw（pcm）

可选值：`raw` / `opus`



**rate** `int`

指定音频采样率，默认值为 `16000`



**bits** `int`

指定音频采样点位数，默认值为`16`



**channel** `int`

指定音频声道数，默认值为 `1`

可选值：

`1`：mono

`2`：stereo




**request** `object`


**model_name** `string` <span data-api-tag="require|tXLKeG">必选</span>

指定模型名称，目前仅支持 `bigmodel`



**enable_speaker_info** `bool`

启用说话人分离参数，默认为`false`

<div data-tips="true" data-tips-type="default">说明</div>



* <div data-tips="true" data-tips-type="default">开启后需启用<code>show_utterances</code>参数，才能获取到说话人分离结果</div>


* <div data-tips="true" data-tips-type="default">仅在 <code>language</code> 未指定，或指定为<code> zh-CN</code> 时生效</div>




**ssd_version** `string`

指定说话人分离场景对应的模型版本，可选值如下：


* `200`：

   * 说话人数量建议不超过5人

   * 适用于**非会议场景**

   * 需将`enable_speaker_info`和`show_utterances`设置为`true`

* `300`：

   * 使用声纹匹配能力

   * 适用于长音频会议场景，包括线上会议、录音笔、录音卡、笔记 App 等**多人说话场景**

   * 需将`enable_speaker_info`和`show_utterances`设置为`true`



**ssd_mode** `int`

指定说话人分离模式，仅`ssd_version：200`模型支持该参数。可选值如下：


* `0`:普通模式（默认），适用于3分钟以内、说话人数小于5的短音频

* `1`:聚类模式，适用于3分钟以上的长音频，如售车、售房、一对多销售等非会议场景



**enable_itn** `bool`

启用将语音识别结果转换为规范的书面格式，默认为`true`

开启后，系统会将语音里口语化的数字、金额及日期等自动转成阿拉伯数字和符号形式，使文本更简洁、更易读

效果示例:


* "一九七零年" → "1970 年"

* "一百二十三美元" → "$123"



**enable_punc** `bool`

启用标点，默认为`true`

开启后，系统会在识别结果中添加逗号、句号、问号等标点符号，提升文本可读性



**enable_ddc** `bool`

启用语义顺滑，默认为 `false`

开启后，系统会删除或修正识别结果中的停顿词、语气词、语义重复词等不流畅内容，让文本更连贯、更易读



**enable_channel_split** `bool`

启用双声道识别，默认为`false`

开启后，返回结果中将以 `channel_id` 标记声道

`1` :左声道

`2` :右声道



**show_utterances** `bool`

启用输出分句、分词、说话人及语音停顿信息，默认为`false`



**show_speech_rate** `bool`

启用分句信息携带语速，默认为`false`

开启后，系统将在分句 `additions` 中返回语速信息，单位为 token/s



**show_volume** `bool`

启用分句信息携带音量，默认 `false`

开启后，系统将在分句 `additions` 中返回音量信息，单位为dB



**enable_auto_lang** `bool`

启用自动识别语种，默认 `false`

开启后，系统会自动检测音频所属语种。支持自动识别以下语种：


* 中文普通话 `zh-CN`

* 英语：`en-US`

* 日语：`ja-JP`

* 印尼语：`id-ID`

* 西班牙语：`es-MX`

* 葡萄牙语：`pt-BR`

* 德语：`de-DE`

* 法语：`fr-FR`

* 韩语：`ko-KR`

* 菲律宾语：`fil-PH`

* 马来语：`ms-MY`

* 泰语：`th-TH`

* 阿拉伯语 `ar-SA`

* 意大利语 `it-IT`

* 孟加拉语 `bn-BD`

* 希腊语 `el-GR`

* 荷兰语 `nl-NL`

* 俄语 `ru-RU`

* 土耳其语 `tr-TR`

* 越南语 `vi-VN`

* 波兰语 `pl-PL`

* 罗马尼亚语 `ro-RO`

* 尼泊尔语 `ne-NP`

* 乌克兰语 `uk-UA`

* 粤语 `yue-CN`



**enable_lid** `bool`

启用中英文及方言识别，默认 `false`

支持识别以下语种：中文、英文、上海话、闽南话、四川话、陕西话、粤语

开启后，系统将在 `additions` 中返回语种/场景标签，取值如下：


* `singing_en`：英文唱歌

* `singing_mand`：普通话唱歌

* `singing_dia_cant`：粤语唱歌

* `speech_en`：英文说话

* `speech_mand`：普通话说话

* `speech_dia_nan`：闽南语

* `speech_dia_wuu`：吴语（含上海话）

* `speech_dia_cant`：粤语说话

* `speech_dia_xina`：西南官话（含四川话）

* `speech_dia_zgyu`：中原官话（含陕西话）

* `other_langs`：其它语种（其它语种人声）

* `others`：检测不出（非语义人声和非人声）

* 返回为空则代表无法判断（例如传入音频过短等）



**enable_emotion_detection** `bool`

启用情绪检测，默认为 `False`

开启后，系统将在分句`additions`中返回对应的情绪标签。支持的情绪标签如下：


* `angry`：表示情绪为生气

* `happy`：表示情绪为开心

* `neutral`：表示情绪为平静或中性

* `sad`：表示情绪为悲伤

* `surprise`：表示情绪为惊讶



**enable_gender_detection** `bool`

启用性别检测，默认为 `False`

开启后，系统将在分句`additions`中返回性别标签（male/female）



**enable_age_detection**`bool`

启用年龄检测，默认为`False`

开启后，系统将在分句`additions`中返回说话人的年龄（age），返回值为字符串类型的浮点数

<div data-tips="true" data-tips-type="default" data-tips-is-title="true">说明</div>


<div data-tips="true" data-tips-type="default">年龄检测是模型基于语音特征对说话人年龄做出的估算，结果仅供参考。实际准确率受录音质量、说话语速、及口音等多种因素影响。</div>




**vad_segment** `bool`

语音活动检测 (VAD) 分句参数，默认为`false`。开启后系统将根据VAD规则进行分句，否则根据语义进行分句

<div data-tips="true" data-tips-type="default" data-tips-is-title="true">说明</div>



* <div data-tips="true" data-tips-type="default">当<code>enable_channel_split</code>设置为<code>true</code>时，建议使用语义分句，即将<code>vad_segment</code>设置为<code>false</code></div>


* <div data-tips="true" data-tips-type="default">若同时配置了<code>end_window_size</code>，此参数不生效</div>







**end_window_size** `int`

语音活动检测 (VAD) 的静音判停阈值，单位 ms。当检测到的连续静音时长达到该值时，判定一句话结束并触发分句。默认值为 `800`

范围：`[300,5000] `

推荐值：`[800,1000]`



**sensitive_words_filter** `string`

启用敏感词过滤功能。开启后，可对识别结果中的敏感词做屏蔽或替换处理

示例

```Bash
"sensitive_words_filter":{\"system_reserved_filter\":true,\"filter_with_empty\":[\"敏感词\"],\"filter_with_signed\":[\"敏感词\"]}"
```



**system_reserved_filter ** `bool`

启用系统内置敏感词库，默认为`false`，启用后，命中的系统敏感词会被替换为 `*`



**filter_with_empty ** `string`

设置需替换为空字符串的自定义敏感词列表



**filter_with_signed ** `string`

设置需替换为 `*` 的自定义敏感词列表




**enable_poi_fc** `bool`

启用 POI Function Call，默认为`false`，启用后可调用专业的地图领域推荐词服务辅助识别，提高识别准确率

示例：

```SQL
"request": {
    "enable_poi_fc": true,
    "corpus": {
        "context": "{\"loc_info\":{\"city_name\":\"北京市\"}}"
    }
}
```




**enable_music_fc** `bool`

启用 Music Function Call，默认为`false`，开启后，对于语音识别困难的词语，能调用专业的音乐领域推荐词服务辅助识别



**corpus** `object`

配置语境词典，可自定义配置热词、替换词和上下文信息，配置后可提高特定语境下的词语识别准确率

<div data-tips="true" data-tips-type="default" data-tips-is-title="true">说明</div>



* <div data-tips="true" data-tips-type="default">使用该能力时，不支持同时设置<code>enable_auto_lang</code>参数</div>


* <div data-tips="true" data-tips-type="default">热词总长度不得超过5000词（含<code>context</code>直传热词与热词表热词）。当热词总长度超过5000词时，系统按热词传入顺序从前向后截断，仅前5000热词生效</div>


* <div data-tips="true" data-tips-type="default">热词属于提示性参数，用于引导模型优先识别特定词汇，但并非强制约束。模型会在识别过程中优先考虑热词，但受语音清晰度、语境等多种因素影响，不保证所有热词都能 100% 正确转写</div>


* <div data-tips="true" data-tips-type="default">识别效果与热词的选词策略、上下文信息有关，为达到最佳效果建议根据<a href="https://docs.volcengine.com/docs/6561/2604976?lang=zh">热词与上下文最佳实践</a>配置热词和上下文</div>




**boosting_table_name ** `string`

热词词表名称，可在[控制台>自学习平台](https://console.volcengine.com/speech/new/hot-word?projectName=default)配置热词后获取



**boosting_table_id ** `string`

热词词表id，可在[控制台>自学习平台](https://console.volcengine.com/speech/new/hot-word?projectName=default)配置热词后获取

<div data-tips="true" data-tips-type="default" data-tips-is-title="true">说明</div>


<div data-tips="true" data-tips-type="default">若传入的<code>boosting_table_name</code>和<code>boosting_table_id</code>对应的热词词表不一致，则以<code>boosting_table_id</code>为准</div>




**correct_table_name ** `string`

替换词词表名称，可在[控制台>自学习平台](https://console.volcengine.com/speech/new/correct-word?projectName=default)配置替换词后获取。配置后，可将模型识别出的特定词汇替换为目标词汇



**correct_table_id ** `string`

替换词词表id，可在[控制台>自学习平台](https://console.volcengine.com/speech/new/correct-word?projectName=default)配置替换词后获取。配置后，可将模型识别出的特定词汇替换为目标词汇

<div data-tips="true" data-tips-type="default" data-tips-is-title="true">说明</div>


<div data-tips="true" data-tips-type="default">若传入的<code>correct_table_name</code>和<code>correct_table_id</code>对应的热词词表不一致，则以<code>correct_table_id</code>为准</div>




**regex_correct_table_name**`string`

正则替换词表名称，可在[控制台>自学习平台](https://console.volcengine.com/speech/new/correct-word?projectName=default)配置正则替换词后获取。相较于替换词的精确匹配替换，正则替换词适合批量格式转换（如日期格式统一、符号标准化）、模糊模式匹配等复杂场景



**regex_correct_table_id ** `string`

正则替换词表id，可在[控制台>自学习平台](https://console.volcengine.com/speech/new/correct-word?projectName=default)配置正则替换词后获取。相较于替换词的精确匹配替换，正则替换词适合批量格式转换（如日期格式统一、符号标准化）、模糊模式匹配等复杂场景



**context ** `string`

上下文功能。在识别请求中传入辅助上下文信息，帮助模型结合语境提升识别准确率。支持传入热词、对话历史、场景描述等多种类型的上下文信息

<div data-tips="true" data-tips-type="default" data-tips-is-title="true">说明</div>



* <div data-tips="true" data-tips-type="default">使用该能力时，不支持同时设置<code>enable_auto_lang</code>参数</div>


* <div data-tips="true" data-tips-type="default">实际传参时<strong> </strong><strong><code>context</code></strong><strong> 需序列化为 JSON 字符串传入</strong>，如：<code>"context": "{\"hotwords\":[{\"word\":\"自定义热词A\"},,\"context_type\":\"dialog_ctx\",\"context_data\":[{\"speaker\":\"bot\",\"text\":\"最近一轮助手的回答\"}"</code></div>


* <div data-tips="true" data-tips-type="default">上下文最大支持上传500 tokens；超出限制时，按时间顺序保留最新内容，截断最早的历史对话</div>


* <div data-tips="true" data-tips-type="default">识别效果与热词的选词策略、上下文信息有关，为达到最佳效果建议根据<a href="https://docs.volcengine.com/docs/6561/2604976?lang=zh">热词与上下文最佳实践</a>配置热词和上下文</div>



示例：

```Python
{
  "hotwords": [
    { "word": "自定义热词A" },
    { "word": "自定义热词B" },
    { "word": "自定义热词C" }
  ],
  "context_type": "dialog_ctx",
  "context_data": [
    {
      "speaker": "user",
      "text": "你能帮我查一下资料吗？"
    },
    {
      "speaker": "bot",
      "text": "当然可以，请问您需要查什么资料？"
    },
    {
      "speaker": "user",
      "text": "帮我查一下最新的汽车资讯。"
    },
    {
      "speaker": "bot",
      "text": "好的，正在为您查找相关的汽车资讯。"
    }
  ]
}
```



**hotwords ** `string`

热词列表直传，用于提升指定词汇的识别准确率


**word** `string`

热词内容

<div data-tips="true" data-tips-type="default" data-tips-is-title="true">说明</div>



* <div data-tips="true" data-tips-type="default">热词属于提示性参数，用于引导模型优先识别特定词汇，但并非强制约束。模型会在识别过程中优先考虑热词，但受语音清晰度、语境等多种因素影响，不保证所有热词都能 100% 正确转写</div>


* <div data-tips="true" data-tips-type="default">识别效果与热词的选词策略、上下文信息有关，为达到最佳效果建议根据<a href="https://docs.volcengine.com/docs/6561/2604976?lang=zh">热词与上下文最佳实践</a>配置热词</div>





**context_type** `string`

上下文类型，目前仅支持`dialog_ctx`



**context_data** `object`

上下文数据列表，用于传入历史对话等语境信息，需同时配置`context_type`


**text ** `string`

历史对话文本，帮助模型理解语境，提升识别准确率



**image_url ** `string`

图片 URL，用于提供视觉上下文，辅助理解语音内容

<div data-tips="true" data-tips-type="default" data-tips-is-title="true">说明</div>


<div data-tips="true" data-tips-type="default">仅豆包录音文件识别模型 2.0 支持图片输入。当前限制：最多传入 1 张图片，单张大小不超过 500 KB，支持格式为<code> jpeg </code>/<code>jpg</code>/<code>png</code></div>








**callback  ** `string`

指定回调地址

示例：

```Python
"callback": "http://xxx"
```




**callback_data** `string`

指定回调信息

```Python
"callback_data":"$Request-Id"
```





<span id="pF6mxalL"></span>
### 响应


**task_id ** `string`

任务 ID，可通过该 ID 调用识别结果查询接口获取识别结果



**X\-Tt\-Logid ** `string`

服务端返回的 logid，方便定位问题



**X\-Api\-Status\-Code ** `string`

提交任务后服务端返回的状态码



**X\-Api\-Message ** `string`

提交任务后服务端返回的信息，`OK` 表示成功，其他值表示失败





