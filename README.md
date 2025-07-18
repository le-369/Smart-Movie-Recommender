# 🎥 影智导航系统

影智导航系统是一个基于 MovieLens 32M 数据集开发的电影推荐系统，旨在通过模块化的代码结构和清晰的文件命名，帮助开发者快速理解推荐系统原理并加速项目部署。系统集成了数据探索、预处理、推荐算法训练、内容相似性计算和 Flask 驱动的 Web 前端，同时提供基于 Scrapy 的 TMDB 爬虫以增强电影信息展示。

## 📌 项目简介

本项目基于 [MovieLens 32M 数据集](https://grouplens.org/datasets/movielens/)，通过模块化设计实现了电影推荐系统的核心功能，包括数据预处理、推荐模型训练和前端展示。系统预留了扩展接口，方便开发者学习、修改或集成新功能。TMDB 爬虫模块用于获取电影元数据以丰富前端界面，尽管直接调用电影网站 API 的集成尚未解决，欢迎社区贡献更优解决方案。

## 📂 项目结构

```bash
├── data/                  # 存储处理后的数据集
├── ml-32m/                # MovieLens 32M 数据集存放目录（需手动下载）
├── reports/               # 算法性能、灵敏度分析、Web 响应时间等图片
├── static/                # 前端静态文件（背景图片、系统框架图）
├── templates/             # 前端 HTML 模板（index.html）
├── TMDB/                  # TMDB 爬虫相关代码
│   ├── movies_tmdb.json   # 爬取的电影数据（需手动移动到根目录）
│   ├── items.py           # 爬虫配置（如 USER_AGENT）
├── 1.explore_data.ipynb   # 数据探索和分析
├── 2.preprocess_movies.py # 数据预处理（缺失值填充）
├── 3.recommendation_system.py # 推荐系统网络训练
├── 4.compute_content_sim.py   # 预计算内容相似性和模型
├── 5.app.py               # Flask Web 前端应用
├── 6.sensitivity_analysis.py  # 灵敏度分析
└── README.md              # 项目文档
```

## 🖼️ 系统框架

下图展示了影智导航系统的整体架构：

<div align="center">
  <img src="static/system.bmp" alt="系统框架" style="width:400px; height:auto;" />
</div>



前端界面设计如下：

![前端构图](static/web.bmp "前端构图")

## 📊 数据集

- **MovieLens 32M**：
  - 下载地址：[MovieLens 32M](https://grouplens.org/datasets/movielens/)
  - 请下载数据集并解压到 `ml-32m` 文件夹（根目录下，不要包含子文件夹）。
- **TMDB 数据**：
  - 通过 `TMDB` 文件夹中的 Scrapy 爬虫获取，存储在 `TMDB/movies_tmdb.json`。
  - 使用后需手动移动到根目录下的 `movies_tmdb.json`。

## 🚀 使用说明

系统通过清晰的文件命名（`1.explore_data.ipynb` 到 `6.sensitivity_analysis.py`）引导用户按顺序运行，逐步理解和部署推荐系统：

1. **`1.explore_data.ipynb`**：数据探索，分析 MovieLens 数据集的特征和分布。
2 **`2.preprocess_movies.py`**：处理缺失值，清洗和格式化数据。
3. **`3.recommendation_system.py`**：训练推荐系统模型。
4. **`4.compute_content_sim.py`**：预计算内容相似性和模型，加速 Web 部署。
5. **`5.app.py`**：启动 Flask Web 服务器，展示推荐结果。
6. **`6.sensitivity_analysis.py`**：执行灵敏度分析，生成性能报告（存储在 `reports` 文件夹）。

### TMDB 爬虫

爬虫用于获取 TMDB 电影元数据以丰富前端展示：

1. 配置爬虫：
   - 编辑 `TMDB/TMDB/items.py`，设置 `USER_AGENT`（参考 [配置教程](https://blog.csdn.net/BobYuan888/article/details/88950275)）。
2. 运行爬虫：
   ```bash
   cd TMDB
   scrapy crawl tmdb
   ```
3. 移动数据：
   - 将生成的 `TMDB/movies_tmdb.json` 手动移动到根目录下的 `movies_tmdb.json`。

**建议**：
- 建议先学习爬虫原理（参考 [Scrapy 教程](https://blog.csdn.net/2301_77659011/article/details/135630678)）。
- 当前通过爬虫获取数据较为耗时，直接调用 TMDB API 会更高效，但本项目未解决 API 集成问题，欢迎社区优化。

## 🎯 项目目标

本项目旨在通过简单的 Demo 帮助开发者：
- **自主学习**：通过模块化代码和命名清晰的文件，快速理解推荐系统的核心原理。
- **加速部署**：提供预处理、训练和 Web 部署的完整流程，方便集成到实际项目。

## 📮 联系方式

如有问题、反馈或合作意向，请联系：
📧 [HELLOLE_369@126.com]