<p align="center">
  <a href="https://github.com/actions/typescript-action/actions"><img alt="typescript-action status" src="https://github.com/actions/typescript-action/workflows/build-test/badge.svg"></a>
</p>

# 🤖 Automated Code Reviews powered by ChatGPT 🤖

A GitHub action uses OpenAI's GPT-4 to perform automated code reviews. When you create a PR, our action will automatically review the code and suggest changes, just like a human code reviewer would. 

## 🚀 How to use it

- Get an API Key from [OpenAI](https://platform.openai.com/account/api-keys)
- Add it as a Github secret
- Setup an action that runs on every PR

```
name: 'code-review'
on: # rebuild any PRs and main branch changes
  pull_request:
jobs:
  code-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: ./
        env:
          NODE_OPTIONS: '--experimental-fetch'
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          openai_api_key: ${{ secrets.OPENAI_API_KEY }}
          exclude_files: '*.js, *.json, *.md, *.yml' # optionally exclude files based on a wildcard expression. 
```
- Or when a label is added
```
name: 'code-review'
on: # rebuild any PRs and main branch changes
  pull_request:
    types: [labeled]
jobs:
  code-review:
    if: ${{ contains( github.event.label.name, 'AutoReview') }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: ./
        env:
          NODE_OPTIONS: '--experimental-fetch'
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          openai_api_key: ${{ secrets.OPENAI_API_KEY }}
          exclude_files: '*.js, *.json, *.md, *.yml' # optionally exclude files based on a wildcard expression. 
```

## Input parameters

| **Parameter**     | **Required** | **Default**   | **Description**                                                                                                                                                                         |
|-------------------|--------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| github_token      | True         |               | Necessary for communicating with Github. Autogenerated by the GHA                                                                                                                       |
| openai_api_key    | True         |               | OpenAI API key                                                                                                                                                                          |
| model_name        | False        | gpt-3.5-turbo | OpenAI ChatModel. Currently supports `gpt-4` and `gpt-3.5-turbo`                                                                                                                        |
| model_temperature | False        | 0             | OpenAI model temperature                                                                                                                                                                |
| exclude_files     | False        |               | Provide a wildcard expression to exclude files from code review.  For example, `*.md` will exclude all markdown files. Multiple  expressions are supported via commas, eg `*.js, *.cjs` |

## 🎉 Benefits

Using our GitHub action has many benefits, such as:
- Faster code reviews
- More consistent feedback
- Increased productivity
- Improved code quality

## 🤞 Limitations

- This Github Action is still in early development. 
- While the action supports both `gpt-4` and `gpt-3.5-turbo`, `gpt-4` gives much better suggestions.
## 🙌 Contributing

If you have any ideas or improvements to our GitHub action, feel free to submit a PR. We welcome all contributions! 
