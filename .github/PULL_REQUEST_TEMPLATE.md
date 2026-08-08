## O que muda

<!-- E porquê. Se corrige um bug, o comportamento errado concreto. -->

## Verificação

- [ ] `ruff check . && ruff format --check .`
- [ ] `pytest -q`
- [ ] Teste de regressão para o comportamento corrigido (se for um fix)

<!--
Sobre testes: este projeto trata medições como factos. Se mudas números de
admissão ou de calibração, diz como os verificaste — de preferência com uma
corrida real de `vramd calibrate` antes/depois.
-->
