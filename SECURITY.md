# Segurança

## Modelo de ameaça

O `vramd` foi feito para correr **localmente**, para um utilizador:

- o socket é um Unix socket com as permissões do utilizador — **não há
  autenticação**;
- qualquer processo desse utilizador pode submeter jobs, carregar backends e
  parar o supervisor;
- um descriptor de backend define **comandos que serão executados**
  (`runtime.command`). Tratar um `backends.yaml` de terceiros com o mesmo
  cuidado com que se trata um script.

Não expor o socket na rede nem partilhá-lo entre utilizadores sem uma camada de
autenticação à frente. Um gateway HTTP exigiria repensar isto de raiz.

## Reportar

Vulnerabilidades: abrir um [security advisory][adv] privado. Para tudo o resto,
uma issue normal.

[adv]: https://github.com/maikramer/vramd/security/advisories/new

Versões suportadas: a última `0.x`. Enquanto o projeto estiver em `0.x`, só a
versão mais recente recebe correções.
