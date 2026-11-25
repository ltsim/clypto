# Configuración de Memoria Neo4j Dedicada para Clypto Development (`clypto-development-memory`)

**Entregable:** `reports/MEMORY-SETUP.md`  
**Destinatarios:** `cy-audit-advisor-code` (`orca_session_id:opencode2_23077bf4_da54_4c4c_a995_ba16ea326f83`), equipo de agentes y subagentes  
**Emisor:** `cy-implementation` / `antigravity` (`orca_session_id:antigravity_3c5a507b_d3ef_41f9_8d4c_84d9c3382509`)  
**Fecha:** 2026-11-11  

---

## 1. Arquitectura y Diseño Físico Dedicado

Neo4j Community Edition (5.26.x) opera sobre una única base de datos de usuario por omisión. Para garantizar un **aislamiento físico absoluto (100%)** —sin heurísticas de proxy, sin fugas salientes hacia `iris-code` ni `factory56`, y sin contaminación entrante desde otros repositorios—, se desplegó un contenedor Docker dedicado exclusivo para el desarrollo y auditoría de Clypto:

- **Nombre del contenedor**: `clypto-development-memory` (imagen `neo4j:5`)
- **Puertos locales**: `127.0.0.1:7690->7687` (Bolt), `127.0.0.1:7476->7474` (HTTP)
- **Volumen persistente**: `clypto-development-memory-data:/data`
- **Servidor MCP**: `neo4j-agent-memory mcp serve` nativo apuntando directo a `bolt://127.0.0.1:7690` (**sin proxy intermediario**).

---

## 2. Archivos de Configuración y Rutas Exactas

### A. Credenciales y Lanzador MCP Dedicado
- **Credenciales**: `/home/deal/.config/clypto-neo4j/.env` (permisos `600`, fuera del repositorio git):
  ```env
  NEO4J_URI="bolt://127.0.0.1:7690"
  NEO4J_USERNAME="neo4j"
  NEO4J_PASSWORD="<password-generada-hex>"
  ```
- **Lanzador MCP**: `/home/deal/.config/clypto-neo4j/serve.sh` (permisos `755`):
  ```sh
  #!/bin/sh
  # Neo4j agent memory MCP dedicated to Clypto Development (Dedicated Instance on 7690: clypto-development-memory)
  set -a
  . "/home/deal/.config/clypto-neo4j/.env"
  set +a
  export UV_INDEX=https://download.pytorch.org/whl/cpu UV_INDEX_STRATEGY=unsafe-best-match
  exec uvx --from "neo4j-agent-memory[mcp,sentence-transformers]" neo4j-agent-memory mcp serve \
    --uri "$NEO4J_URI" --user "$NEO4J_USERNAME" --password "$NEO4J_PASSWORD" \
    --profile core --session-strategy persistent --user-id clypto-development \
    --embedding sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --embedding-dimensions 384
  ```

### B. OpenCode (`cy-audit-advisor-code` y subagentes)
- **Ruta**: `/home/deal/orca/workspaces/clypto/audit-simplifications/.opencode/opencode.json`
- **Configuración (entrada única MCP)**:
  ```json
  {
    "$schema": "https://opencode.ai/config.json",
    "model": "opencode/deepseek-v4.1-flash-free",
    "default_agent": "advisor",
    "mcp": {
      "clypto-development-memory": {
        "type": "local",
        "command": ["bash", "/home/deal/.config/clypto-neo4j/serve.sh"],
        "enabled": true
      }
    }
  }
  ```

### C. Antigravity (`cy-implementation`)
- **Ruta**: `~/.gemini/config/mcp_config.json`
- **Configuración (entrada única MCP)**:
  ```json
  {
    "mcpServers": {
      "clypto-development-memory": {
        "command": "/home/deal/.config/clypto-neo4j/serve.sh",
        "args": []
      }
    }
  }
  ```

---

## 3. Evidencias de Verificación y Aceptación

### A. Estado del Contenedor y Servicio Bolt (7690)
```text
$ docker ps --filter "name=clypto-development-memory"
CONTAINER ID   IMAGE     COMMAND                  CREATED         STATUS         PORTS                                                          NAMES
61cccc2a267b   neo4j:5   "tini -g -- /startup…"   5 minutes ago   Up 5 minutes   7473/tcp, 127.0.0.1:7476->7474/tcp, 127.0.0.1:7690->7687/tcp   clypto-development-memory

$ python3 -c "SHOW DATABASES on bolt://127.0.0.1:7690"
Databases online: ['neo4j', 'system']
```

### B. Pruebas MCP en la Instancia Dedicada (7690)
```text
clypto-development memory_search('benchmark') total messages: 0
clypto-development memory_search('clypto release v2026b1') total messages: 0
clypto-development memory_add_fact result: {"stored": true, "type": "fact", "id": "24caa33b-356e-4f42-95d4-788c9a78c724", "triple": "clypto-development -> isolation_test -> physical_db_verified"}
clypto-development memory_store_message result: {"stored": true, "type": "message", "id": "aef30734-f699-497c-846f-d27cd80dbd5b", "session_id": "clypto-development"}
clypto-development memory_search('clypto-development persistent memory isolation verified') messages found: 1
  found: clypto-development persistent memory isolation verified
```

### C. Verificación de Cero Fuga hacia `iris-code` / `factory56` (7688)
```text
$ python3 verify_cross_db.py
Count of message 'aef30734-f699-497c-846f-d27cd80dbd5b' in bolt://127.0.0.1:7688 (iris-memory-neo4j): 0
Count of message 'aef30734-f699-497c-846f-d27cd80dbd5b' in bolt://127.0.0.1:7690 (clypto-development-memory): 1

iris-code (7688) memory_search('benchmark') total messages: 2 (datos de factory56 intactos)
```

---

## 4. Procedimiento de Rollback y Desmantelamiento

Para desmantelar completamente la instancia y su volumen sin impactar otros entornos:

```bash
# 1. Detener y remover el contenedor dedicado
docker rm -f clypto-development-memory

# 2. Eliminar el volumen persistente
docker volume rm clypto-development-memory-data

# 3. Restaurar configuraciones MCP previas si se desea volver a la compartida:
# En ~/.gemini/config/mcp_config.json y opencode.json apuntar a /home/deal/.config/iris-neo4j/serve.sh
```
