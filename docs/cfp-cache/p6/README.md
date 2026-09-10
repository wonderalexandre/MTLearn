# CFP cache — conclusão da P6

Data: 2026-09-09. **P0–P6 concluídas no escopo aprovado.**

A CFP oferece preparação sem retenção, RAM limitada e SSD local com retomada e processos CPU. Os scorers podem compartilhar dados brutos e manter parâmetros e estatísticas próprios. A interface legada e os checkpoints anteriores continuam funcionando. O notebook específico foi validado em resolução original na P5.

**619 testes aprovados**, incluindo **19 gradchecks** e **24 novos casos de checkpoints sem armazenamento de preparação**, com zero falhas, erros ou skips. O wheel final, construído a partir do pacote fonte e instalado em uma pasta isolada, passou no smoke test das APIs públicas e nos mesmos 24 casos de checkpoints CPU/MPS. A documentação passou em modo estrito com imports reais e com dependências pesadas simuladas como ausentes.

## Escopo da documentação

O propósito dos guias existentes foi avaliado antes das alterações:

| Documento | Alteração nesta etapa |
| --- | --- |
| [Migração do cache](../../source/guides/cfp-cache-migration.md) | Página nova: escolha de política, três caminhos de migração, propriedade dos buffers, cotas, identidade, retomada, paralelismo e checkpoints. |
| [Referência de preparação](../../source/api/python/cfp_preparation.rst) | Página nova com dez objetos públicos de preparação, armazenamento e estatísticas. |
| [Arquitetura](../../cfp-architecture.md) | Acrescenta responsabilidades e fluxo de preparação/armazenamento, mantendo as seções de scoring, constraints, regularização, loss e sinal reconstruído. O diagrama existente foi preservado. |
| [Guia da CFP](../../source/guides/connected-filter-preprocessing.md) | Somente três linhas de navegação para a nova página. O conteúdo detalhado das etapas P1–P4 permaneceu. |
| [Integração PyTorch](../../source/guides/pytorch-integration.md) | Corrige a afirmação de que o carregador legado é obrigatório para ajustar estatísticas e esclarece CPU versus dispositivo ativo. Mantém o modelo e o exemplo de treino existentes. |
| Índices dos guias e da API | Incluem as duas páginas novas. |

Nenhum guia de scoring, constraints, regularização, loss ou experimento TIP2026 foi reescrito. O código científico da biblioteca e os notebooks não mudaram nesta etapa.

Os três exemplos da página nova foram extraídos e executados com um dataset pequeno: ajuste sem retenção, RAM limitada e SSD com dois processos CPU, seguido de backward. A inspeção do HTML verificou a presença dos dez objetos públicos e 274 links/âncoras locais em seis páginas com imports reais; a versão com mocks verificou 273 links/âncoras. Evidências: [documentação](docs-validation.json) e [exemplos](guide-examples.log).

## Pacotes e compatibilidade

O build real da distribuição fonte revelou dois problemas de empacotamento, corrigidos apenas em `pyproject.toml`:

- Os alvos opcionais de documentação do backend tentavam configurar arquivos Doxygen excluídos do pacote quando Doxygen estava instalado. `MMCFILTERS_ENABLE_DOC_TARGETS=OFF` desativa esses alvos no build do pacote.
- A exclusão de notebooks cobria somente a pasta da raiz, deixando entrar um `.gitkeep` em uma pasta aninhada. A exclusão passou a ser recursiva, como exige o pipeline existente de distribuição.

O build completo foi repetido após as correções: **sdist → wheel**, sem baixar dependências e com um processo de compilação. O log conserva avisos do CMake sobre a política de descoberta Python e a biblioteca opcional Kineto; eles não impediram a compilação nem os testes. A falha inicial está em [package-build-initial-failure.log](package-build-initial-failure.log); o resultado final está em [package-build.log](package-build.log).

O [manifesto dos pacotes](package-manifest.json) registra nomes, tamanhos e SHA-256. Foram conferidos por conteúdo **78 módulos Python**, incluindo os **20 módulos novos** de preparação, armazenamento, snapshot e lifetime, tanto no wheel quanto no sdist. O arquivo de versão é gerado pelo build e não entra nessa comparação de código. As 25 fixtures imutáveis da P0 estão preservadas no sdist, e as exclusões exigidas pelo pipeline passaram.

O wheel local é `mtlearn-1.1.0.post5-cp312-cp312-macosx_26_0_arm64.whl`, instalado em `build/cfp-cache-p6-install-final`. O teste foi iniciado de `/private/tmp` com `python -I`, removendo somente nesse processo o redirecionador de outro checkout editável. Os caminhos importados do pacote **e da extensão nativa** foram conferidos dentro da instalação isolada. Não foi alterada a instalação de trabalho.

O smoke test de distribuição agora verifica importação dos dez objetos públicos, ajuste estatístico, equivalência de saídas e gradientes em execução direta/RAM/sem retenção/SSD, preparação com um trabalhador CPU, reabertura do SSD e carregamento por `PreparedDataset`/`collate_prepared`.

Os 24 casos adicionais restauram os **12 checkpoints anteriores ao refatoramento em CPU e MPS**, usando as fixtures de saída, loss, gradientes, contratos e estatísticas da P0. Durante todo o teste, abrir ou consultar `DiskStore` e conectar ao SQLite provoca falha. A execução ocorre em uma pasta vazia, que permanece vazia. Assim, carregar um modelo e executar imagens diretamente não depende de um diretório ou manifesto de preparação. Evidências: [teste integrado](validation-tests.xml) e [instalação isolada](installed-wheel/validation.json).

O backend usado pelos testes do checkout manteve o SHA-256 da P0. O wheel recém-compilado possui outro hash de binário, registrado separadamente. Como a identidade do cache é conservadora, uma compilação diferente pode exigir outro diretório de preparação; isso não afeta a portabilidade dos checkpoints.

## Critérios de aceitação e evidências

| Critério | Evidência |
| --- | --- |
| Saídas, gradientes e contratos preservados | Fixtures P0; execução direta, legada, RAM e SSD em CPU/MPS; 19 gradchecks na regressão final. |
| Estatísticas do treino sem ajuste por época | [P1](../p1/README.md), [P3](../p3/README.md) e [P5](../p5/README.md): snapshots, resumos, redução ordenada e proibição diagnóstica de atualizar/reduzir durante o treino. |
| Retenção limitada e lifetime de autograd | [P2](../p2/README.md): contagem por storages, LRU, entradas grandes, evicção e liberação antes/depois do backward. |
| SSD, identidade e retomada | [P3](../p3/README.md): arquivos incompletos/corrompidos, falhas de publicação/registro, cota e contribuição única. |
| Processos e filas limitados | [P4](../p4/README.md): spawn, retries, cancelamento, encerramento abrupto, leases e coordenação das gravações. |
| Compartilhamento linear/MLP e carregadores | P2–P5: mesma preparação com parâmetros distintos, IDs, sampler/Subset e comportamento legado preservado. |
| Notebook em resolução original | [P5](../p5/README.md): três imagens de 2748 × 2748, uma época por scorer e avaliação em lotes. |
| Checkpoints sem SSD e imports distribuídos | P6: 24 casos novos e repetição sobre o wheel final instalado. |
| Documentação e migração gradual | P6: duas páginas novas, ajustes pontuais, builds estritos e exemplos executados. |

## Medições consolidadas

Estas são **medições já realizadas em P0–P5**, não benchmarks novos da P6. Equipamento: Apple M4, 10 CPUs lógicas, 16 GiB de memória unificada. As amostras reais usaram resolução original e os 63 atributos da configuração do experimento.

| Ensaio | Tempo | Pico RSS observado | Observação |
| --- | ---: | ---: | --- |
| P1, ajuste CPU em três imagens | 64,77 s | 1,111 GiB | Nenhuma preparação retida ao final. |
| P2, preparação/consumo CPU com RAM de 256 MiB | 65,74 s | 1,278 GiB | Três preparações, três acertos e duas evicções. |
| P3, leitura e forward/backward dos dois scorers em três imagens | 3,97 s | 1,187 GiB | Dados já preparados; não inclui o custo inicial de construção. |
| P4, preparação com um trabalhador | 59,54 s | 1,695 GiB agregado | Três imagens; frio. |
| P4, preparação com dois trabalhadores | 43,18 s | 2,359 GiB agregado | Mesmas três imagens; frio, 27,5% menos tempo. |
| P4, retomada completa | < 0,8 s | Ver relatório P4 | Não criou trabalhadores nem novas árvores. |
| P5, notebook completo de integração | 96,07 s | 1,664 GiB agregado | Três imagens, uma época por scorer em MPS, incluindo preparação e avaliação. |

P1–P4 usaram IDs 0, 1000 e 2040 nos ensaios de três imagens. A P5 usou treino 1068/1316 e teste 1000, conforme o split do notebook. Os tempos representam fases diferentes e não devem ser comparados como se fossem o mesmo workload. Os arquivos SSD ocuparam aproximadamente 717,56 MiB na P3/P4 e 705,10 MiB na P5, sob cotas de 1 GiB.

A projeção da P0 para o conjunto completo é de **476,71 GiB de tensores brutos**, antes de serialização e margens. A estimativa do cache legado é de 784,73 GiB. SSD reduz retenção de RAM, mas não elimina o armazenamento necessário nem o espaço de trabalho de uma imagem. A biblioteca não adota esses valores como limites universais.

As medidas agregadas de RSS podem contar páginas compartilhadas mais de uma vez e não capturam necessariamente picos menores que o intervalo de amostragem. **RSS não mede todo o pico do driver MPS/memória unificada.** Retenção de cache, dados ativos, filas, workspace nativo e memória do acelerador são controles/medidas distintos.

## Reprodução

A [validação estruturada](validation.json) registra hashes, ambiente, contagens e limites. Os comandos abaixo foram executados; os logs ficam nesta pasta. O ambiente local usa `/opt/anaconda3/bin/python` e a extensão Release da P0. O driver P4 impede que a instalação editável de outro checkout interfira nos testes e nos imports com spawn.

```bash
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p4.py test -- -q mtlearn/tests/python/test_bindings.py mtlearn/tests/python/test_data.py mtlearn/tests/python/test_cfp_preparation.py mtlearn/tests/python/test_cfp_prepared_runtime.py mtlearn/tests/python/test_cfp_disk_cache.py mtlearn/tests/python/test_cfp_parallel_preparation.py mtlearn/tests/python/test_cfp_notebook_streaming.py mtlearn/tests/python/test_cfp_components.py mtlearn/tests/python/test_cfp_deterministic.py mtlearn/tests/python/test_cfp_validation.py mtlearn/tests/python/test_cfp_cache.py mtlearn/tests/python/test_morphology_api.py mtlearn/tests/python/test_gradchecks.py mtlearn/tests/python/test_version.py mtlearn/tests/python/test_cfp_cache_baseline.py mtlearn/tests/python/test_cfp_checkpoint_independence.py --junitxml=docs/cfp-cache/p6/validation-tests.xml
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p6.py docs --output build/cfp-cache-p6-docs
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p6.py docs --mock-runtime --output build/cfp-cache-p6-docs-mocked
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p6.py examples --output build/cfp-cache-p6-examples
CMAKE_BUILD_PARALLEL_LEVEL=1 /opt/anaconda3/bin/python -m build --no-isolation --outdir build/cfp-cache-p6-dist
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p6.py archives --output docs/cfp-cache/p6
/opt/anaconda3/bin/python -m pip install --no-deps --no-index --target build/cfp-cache-p6-install-final build/cfp-cache-p6-dist/*.whl
```

A partir de `/private/tmp`, usando os caminhos deste equipamento:

```bash
/opt/anaconda3/bin/python -I /Users/wonderalexandre/GitHub/mtlearn/scripts/benchmarks/cfp_cache_p6.py wheel --install-root /Users/wonderalexandre/GitHub/mtlearn/build/cfp-cache-p6-install-final --output /Users/wonderalexandre/GitHub/mtlearn/docs/cfp-cache/p6/installed-wheel
```

A conferência de links/âncoras, hashes e evidências pode ser repetida com
`python scripts/benchmarks/cfp_cache_p6_finalize.py` após esses comandos.

Também passaram a compilação sintática dos módulos/testes/scripts alterados, `git diff --check`, a integridade das 25 fixtures P0 e o SHA-256 do notebook original de screws. A regressão final levou 39,16 s; os 24 testes de checkpoints na instalação final levaram 0,91 s, além do smoke test anterior.

## Limites da conclusão

- Validação local em macOS ARM64, Python 3.12.3 e PyTorch 2.10.0, CPU/MPS. CUDA, Linux, Windows e outras versões Python/PyTorch não foram executados. O wheel local não representa uma distribuição multiplataforma.
- O treinamento integral das 2.041 imagens por 100 épocas e a preparação integral do dataset não foram realizados, conforme o escopo aprovado.
- Quatro trabalhadores não foram medidos: a projeção da P4 excedeu o orçamento conservador de 4 GiB. Não há recomendação universal de concorrência.
- A P6 reutiliza os benchmarks e a execução do notebook das etapas anteriores; não altera nem repete esses ensaios como evidência nova.
- Não houve mudança de algoritmos nativos; a validação desta etapa incluiu compilação nativa do wheel e execução Python, sem nova campanha C++/ctest.
- Armazenamento remoto, preparação distribuída, compressão e divisão de uma imagem em patches permanecem fora desta entrega. Uma imagem que exceda a memória de trabalho continua sendo uma limitação explícita.

Os artefatos foram gerados localmente. Não houve publicação externa do pacote ou da documentação.
