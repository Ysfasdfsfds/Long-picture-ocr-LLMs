# Claude Code Context MCP 配置

这个目录包含了为OCR长图智能处理系统配置的Claude Code Context MCP设置。

## 文件说明

### mcp_config.json
MCP服务器配置文件，定义了以下服务器：
- **code-context**: 代码上下文分析服务器
- **filesystem**: 文件系统访问服务器  
- **git**: Git仓库管理服务器

### project_context.md
项目整体上下文描述，包括：
- 项目概述和核心架构
- 主要模块和处理流程
- 技术栈和配置信息
- 代码质量要求

### analysis_config.json
代码分析配置，定义了：
- 项目结构分析规则
- 核心模块分类
- 关键依赖项映射
- 代码模式识别规则

## 使用方法

1. 确保已安装Claude Code CLI
2. 在项目根目录运行：
   ```bash
   claude-code --mcp-config .claude_context/mcp_config.json
   ```

3. 或者将MCP配置添加到全局Claude Code配置中

## 环境变量

需要设置以下环境变量：
- `ANTHROPIC_API_KEY`: Anthropic API密钥
- `PROJECT_ROOT`: 项目根目录路径（已预设为当前项目路径）

## 功能特性

- 自动项目结构分析
- 代码上下文理解
- 智能文件导航
- Git历史分析
- 模块间依赖关系映射