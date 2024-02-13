/* eslint-disable filenames/match-regex */
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate
} from 'langchain/prompts'
import { LLMChain } from 'langchain/chains'
import { BaseChatModel } from 'langchain/dist/chat_models/base'
import type { ChainValues } from 'langchain/dist/schema'
import { PullRequestFile } from './pullRequestService'
import parseDiff from 'parse-diff'
import { LanguageDetectionService } from './languageDetectionService'
export class CodeReviewService {
  private llm: BaseChatModel
  private chatPrompt = ChatPromptTemplate.fromPromptMessages([
    SystemMessagePromptTemplate.fromTemplate(
      "Buď empatický softwarový inženýr, který je odborníkem na všechny programovací jazyky, frameworky a softwarovou architekturu."
    ),
    HumanMessagePromptTemplate.fromTemplate(`Tvým úkolem je zkontrolovat Pull Request. Obdržíš git diff. 
    Zkontrolujte jej a navrhněte případná zlepšení kvality kódu, udržovatelnosti, čitelnosti, výkonu, bezpečnosti atd.
    Identifikuj případné chyby nebo bezpečnostní zranitelnosti (to je důležité). Zkontroluj, zda dodržuje standardy kódování a osvědčené postupy.
    Nenavrhuj přidání komentářů do kódu! Reaguj jen na problémové věci, pokud je něco v pořádku tak nereaguj!
    Odpověď a příklady napiš ve formátu GitHub Markdown. Programovací jazyk je {lang}. Pokud jde o PHP, zkontroluj dodržování
    PSR standardů, php ke kontrole je psané v php 8.3 - takže kontroluj, jestli jsou využity všechny jeho prvky.
    Pokud bude vše v pořádku, vrať odpověď jen a pouze OK, vekými písmeny a bez uvozovek - nic víc!

    git diff na review

    {diff}`)
  ])
  private chain: LLMChain<string>
  private languageDetectionService: LanguageDetectionService

  constructor(
    llm: BaseChatModel,
    languageDetectionService: LanguageDetectionService
  ) {
    this.llm = llm
    this.chain = new LLMChain({
      prompt: this.chatPrompt,
      llm: this.llm
    })
    this.languageDetectionService = languageDetectionService
  }

  async codeReviewFor(file: PullRequestFile): Promise<ChainValues> {
    const programmingLanguage = this.languageDetectionService.detectLanguage(
      file.filename
    )
    return await this.chain.call({
      lang: programmingLanguage,
      diff: file.patch
    })
  }

  async codeReviewForChunks(file: PullRequestFile): Promise<ChainValues> {
    const programmingLanguage = this.languageDetectionService.detectLanguage(
      file.filename
    )
    const fileDiff = parseDiff(file.patch)[0]
    const chunkReviews: ChainValues[] = []

    for (const chunk of fileDiff.chunks) {
      if (chunk.content != 'OK') {
        const chunkReview = await this.chain.call({
          lang: programmingLanguage,
          diff: chunk.content
        })

        chunkReviews.push(chunkReview)
      }
    }

    return chunkReviews
  }
}
