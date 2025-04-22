# Roo Commander Mode Index (v7.1)

*Generated on: 2025-04-17T05:31:16.705Z*

**Total Modes Found:** 88

## Assistant

### Domain: context

*   **[⏱️ Session Summarizer (`session-summarizer`)](v7.1/modes/assistant/context/session-summarizer/session-summarizer.mode.md)**
    *   *Summary:* Reads project state artifacts (task logs, plans) to generate a concise handover summary.
*   **[🌐 Research & Context Builder (`research-context-builder`)](v7.1/modes/assistant/context/research-context-builder/research-context-builder.mode.md)**
    *   *Summary:* Researches topics using web sources, code repositories, and local files, evaluates sources, gathers data, and synthesizes findings into structured summaries with citations.
*   **[🔍 Discovery Agent (`discovery-agent`)](v7.1/modes/assistant/context/discovery-agent/discovery-agent.mode.md)**
    *   *Summary:* Analyzes project context, interacts with users to gather requirements, detects the technical stack, and produces a discovery report.

### Domain: knowledge-management

#### Sub-Domain: summarization

*   **[🧠 Context Condenser (`context-condenser`)](v7.1/modes/assistant/utility/context-condenser/context-condenser.mode.md)**
    *   *Summary:* Generates dense, structured summaries (Condensed Context Indices) from technical documentation sources for embedding into other modes' instructions.

### Domain: utility

*   **[📖 Context Resolver (`context-resolver`)](v7.1/modes/assistant/utility/context-resolver/context-resolver.mode.md)**
    *   *Summary:* Specialist in reading project documentation (task logs, decision records, planning files) to provide concise, accurate summaries of the current project state. Acts as the primary information retrieval and synthesis service for other modes.
*   **[🔧 File Repair Specialist (`file-repair-specialist`)](v7.1/modes/assistant/utility/file-repair-specialist/file-repair-specialist.mode.md)**
    *   *Summary:* Attempts to fix corrupted or malformed text files (such as source code, JSON, YAML, configs) by addressing common issues like encoding errors, basic syntax problems, truncation, and invalid characters.
*   **[🕷️ Crawl4AI Specialist (`crawl4ai-specialist`)](v7.1/modes/assistant/utility/crawl4ai-specialist/crawl4ai-specialist.mode.md)**
    *   *Summary:* <<< ADD SUMMARY >>>

### Domain: web-scraping

*   **[🔥 Firecrawl Specialist (`firecrawl-specialist`)](v7.1/modes/assistant/web-scraping/firecrawl-specialist/firecrawl-specialist.mode.md)**
    *   *Summary:* Implements web crawling and content extraction solutions using the Firecrawl service/API, focusing on configuration, job management, and data retrieval.

## Director

### Domain: product

*   **[🗺️ Product Manager (`product-manager`)](v7.1/modes/director/product/product-manager/product-manager.mode.md)**
    *   *Summary:* A strategic director-level mode responsible for defining and executing product vision, strategy, and roadmap. Translates business goals and user needs into actionable product requirements, coordinates with technical teams, and ensures product success through data-driven decision making.

### Domain: project

*   **[📋 Project Manager (MDTM) (`project-manager`)](v7.1/modes/director/project/project-manager/project-manager.mode.md)**
    *   *Summary:* Manages project features/phases using the TOML-based Markdown-Driven Task Management (MDTM) system, breaking down work, delegating tasks, tracking status, and reporting progress. Operates primarily within the `.tasks/` directory.
*   **[🚦 Project Onboarding (`project-onboarding`)](v7.1/modes/director/project/project-onboarding/project-onboarding.mode.md)**
    *   *Summary:* Handles initial user interaction, determines project scope (new/existing), delegates discovery/requirements gathering, coordinates basic setup, and delegates tech initialization.

### Domain: technical

*   **[🏗️ Technical Architect (`technical-architect`)](v7.1/modes/director/technical/technical-architect/technical-architect.mode.md)**
    *   *Summary:* Designs and oversees high-level system architecture, making strategic technology decisions that align with business goals and technical requirements. Responsible for establishing the technical vision, selecting appropriate technologies, evaluating architectural trade-offs, addressing non-functional requirements, and ensuring technical coherence across the project. Acts as the primary technical decision-maker and advisor for complex system design challenges.

## Executive

### Domain: core

*   **[👑 Roo Commander (`roo-commander`)](v7.1/modes/executive/roo-commander/roo-commander.mode.md)**
    *   *Summary:* Highest-level coordinator for software development projects, managing goals, delegation, and project state.

## Footgun

### Domain: core

*   **[⚡️ Footgun Code (`footgun-code`)](v7.1/modes/footgun/core/footgun-code/footgun-code.mode.md)**
    *   *Summary:* <<< ADD SUMMARY >>>
*   **[📐 Footgun Architect (`footgun-architect`)](v7.1/modes/footgun/core/footgun-architect/footgun-architect.mode.md)**
    *   *Summary:* <<< ADD SUMMARY >>>
*   **[🔬 Footgun Debug (`footgun-debug`)](v7.1/modes/footgun/core/footgun-debug/footgun-debug.mode.md)**
    *   *Summary:* <<< ADD SUMMARY >>>
*   **[🗣️ Footgun Ask (`footgun-ask`)](v7.1/modes/footgun/core/footgun-ask/footgun-ask.mode.md)**
    *   *Summary:* <<< ADD SUMMARY >>>

## Lead

### Domain: backend

*   **[⚙️ Backend Lead (`backend-lead`)](v7.1/modes/lead/backend/backend-lead/backend-lead.mode.md)**
    *   *Summary:* Coordinates backend development (APIs, logic, data integration), manages workers, ensures quality, security, performance, and architectural alignment.

### Domain: database

*   **[💾 Database Lead (`database-lead`)](v7.1/modes/lead/database/database-lead/database-lead.mode.md)**
    *   *Summary:* Coordinates database tasks (schema design, migrations, query optimization, security), manages workers, ensures data integrity and performance.

### Domain: design

*   **[🎨 Design Lead (`design-lead`)](v7.1/modes/lead/design/design-lead/design-lead.mode.md)**
    *   *Summary:* Coordinates design tasks (UI, diagrams), manages design workers, ensures quality and consistency, and reports progress to Directors.

### Domain: devops

*   **[🚀 DevOps Lead (`devops-lead`)](v7.1/modes/lead/devops/devops-lead/devops-lead.mode.md)**
    *   *Summary:* Coordinates DevOps tasks (CI/CD, infra, containers, monitoring, deployment), manages workers, ensures operational stability and efficiency.

#### Sub-Domain: aws

*   **[☁️ AWS Architect (`aws-architect`)](v7.1/modes/lead/devops/aws/aws-architect/aws-architect.mode.md)**
    *   *Summary:* Designs, implements, and manages secure, scalable, and cost-effective AWS infrastructure solutions. Translates requirements into cloud architecture and IaC.

#### Sub-Domain: azure

*   **[☁️ Azure Architect (`azure-architect`)](v7.1/modes/lead/devops/azure/azure-architect/azure-architect.mode.md)**
    *   *Summary:* Specialized Lead for designing, implementing, managing, and optimizing Azure infrastructure solutions using IaC.

#### Sub-Domain: gcp

*   **[☁️ GCP Architect (`gcp-architect`)](v7.1/modes/lead/devops/gcp/gcp-architect/gcp-architect.mode.md)**
    *   *Summary:* A specialized lead-level mode responsible for designing, implementing, and managing secure, scalable, and cost-effective Google Cloud Platform (GCP) infrastructure solutions. Translates high-level requirements into concrete cloud architecture designs and Infrastructure as Code (IaC) implementations.

### Domain: frontend

*   **[🖥️ Frontend Lead (`frontend-lead`)](v7.1/modes/lead/frontend/frontend-lead/frontend-lead.mode.md)**
    *   *Summary:* Coordinates frontend development tasks, manages frontend workers, ensures code quality, performance, and adherence to design/architecture.

### Domain: qa

*   **[🧪 QA Lead (`qa-lead`)](v7.1/modes/lead/qa/qa-lead/qa-lead.mode.md)**
    *   *Summary:* The QA Lead is responsible for coordinating and overseeing all quality assurance activities within the project. They ensure that software releases meet quality standards by planning, delegating, and monitoring testing efforts. They receive features ready for testing or high-level quality objectives from Directors (e.g., Project Manager) or other Leads (e.g., Frontend Lead, Backend Lead) and translate them into actionable testing tasks for the QA Worker modes. Their primary goals are to ensure thorough test coverage, facilitate effective bug detection and reporting, assess product quality, and communicate quality-related risks.

### Domain: security

*   **[🛡️ Security Lead (`security-lead`)](v7.1/modes/lead/security/security-lead/security-lead.mode.md)**
    *   *Summary:* Coordinates security strategy, risk management, compliance, incident response, and manages security specialists.

## Worker

### Domain: ai-ml

*   **[🤗 Hugging Face Specialist (`huggingface-specialist`)](v7.1/modes/worker/ai-ml/huggingface-specialist/huggingface-specialist.mode.md)**
    *   *Summary:* Implements solutions using Hugging Face Hub models and libraries (Transformers, Diffusers, Datasets, etc.) for various AI/ML tasks including natural language processing, computer vision, audio processing, and generative AI. Specializes in model selection, inference implementation, data preprocessing, and integration with application code.
*   **[🧠 OpenAI Specialist (`openai-specialist`)](v7.1/modes/worker/ai-ml/openai-specialist/openai-specialist.mode.md)**
    *   *Summary:* Implements solutions using OpenAI APIs (GPT models, Embeddings, DALL-E, etc.), focusing on prompt engineering and API integration. Specializes in selecting appropriate models, crafting effective prompts, and integrating OpenAI services securely and efficiently into applications.

### Domain: auth

*   **[🔑 Supabase Auth Specialist (`supabase-auth-specialist`)](v7.1/modes/worker/auth/supabase-auth-specialist/supabase-auth-specialist.mode.md)**
    *   *Summary:* Implements and manages user authentication and authorization using Supabase Auth, including RLS policies and frontend integration.
*   **[🔥 Firebase Auth Specialist (`firebase-auth-specialist`)](v7.1/modes/worker/auth/firebase-auth-specialist/firebase-auth-specialist.mode.md)**
    *   *Summary:* Implements and manages user authentication and authorization using Firebase Authentication, including Security Rules and frontend integration. Specializes in configuring Firebase Auth providers, implementing authentication flows, managing user sessions, and defining access control rules within the Firebase ecosystem.

### Domain: backend

*   **[🎯 Directus Specialist (`directus-specialist`)](v7.1/modes/worker/backend/directus-specialist/directus-specialist.mode.md)**
    *   *Summary:* You are Roo Directus Specialist, responsible for implementing sophisticated solutions using the Directus headless CMS (typically v9+).
*   **[🐍 Django Developer (`django-developer`)](v7.1/modes/worker/backend/django-developer/django-developer.mode.md)**
    *   *Summary:* Specializes in building secure, scalable, and maintainable web applications using the high-level Python web framework, Django.
*   **[🐘 PHP/Laravel Developer (`php-laravel-developer`)](v7.1/modes/worker/backend/php-laravel-developer/php-laravel-developer.mode.md)**
    *   *Summary:* Builds and maintains web applications using PHP and the Laravel framework, including Eloquent, Blade, Routing, Middleware, Testing, and Artisan.
*   **[🔌 API Developer (`api-developer`)](v7.1/modes/worker/backend/api-developer/api-developer.mode.md)**
    *   *Summary:* Designs, implements, tests, documents, and secures robust APIs (REST, GraphQL) following best practices.
*   **[🔌 WordPress Specialist (`wordpress-specialist`)](v7.1/modes/worker/backend/wordpress-specialist/wordpress-specialist.mode.md)**
    *   *Summary:* Develops and maintains WordPress sites, including plugin development, theme customization, REST API endpoints, and core functionality, following best practices.
*   **[🔥 Firebase Developer (`firebase-developer`)](v7.1/modes/worker/backend/firebase-developer/firebase-developer.mode.md)**
    *   *Summary:* Expert in designing, building, and managing applications using the comprehensive Firebase platform. Your expertise covers the core suite: Firestore (data modeling, security rules, queries), Authentication (flows, providers, security), Cloud Storage (rules, uploads/downloads), Cloud Functions (triggers, HTTP, callable, Node.js/Python), and Hosting (deployment, configuration). You are proficient with the Firebase CLI (emulators, deployment) and client-side SDKs (especially Web v9 modular SDK). You also have knowledge of other Firebase services like Realtime Database, Remote Config, and Cloud Messaging, along with best practices for cost optimization, testing, and security.
*   **[🧪 Flask Developer (`flask-developer`)](v7.1/modes/worker/backend/flask-developer/flask-developer.mode.md)**
    *   *Summary:* Expert in developing robust web applications and APIs using the Flask Python microframework, including application structuring, extension integration, templating, testing, and security best practices.
*   **[🧱 Supabase Developer (`supabase-developer`)](v7.1/modes/worker/backend/supabase-developer/supabase-developer.mode.md)**
    *   *Summary:* Expert in leveraging the full Supabase platform — including Postgres database, Authentication, Storage, Edge Functions, Realtime, and Vector database — to design, implement, and optimize backend features.
*   **[🚀 FastAPI Developer (`fastapi-developer`)](v7.1/modes/worker/backend/fastapi-developer/fastapi-developer.mode.md)**
    *   *Summary:* Expert in building high-performance APIs with Python using FastAPI, including async operations, Pydantic validation, WebSockets, ORM integration, and testing.
*   **[🛠️ Frappe Specialist (`frappe-specialist`)](v7.1/modes/worker/backend/frappe-specialist/frappe-specialist.mode.md)**
    *   *Summary:* A specialized development mode focused on building and maintaining applications using the Frappe Framework. Expert in database operations, DocType management, schema design, site management, customization, and framework-specific development patterns. Specializes in creating efficient, maintainable Frappe/ERPNext solutions with robust data models.

### Domain: cross-functional

*   **[♻️ Refactor Specialist (`refactor-specialist`)](v7.1/modes/worker/cross-functional/refactor-specialist/refactor-specialist.mode.md)**
    *   *Summary:* You are Roo Refactor Specialist, an expert focused *exclusively* on improving the internal structure, readability, maintainability, and potentially performance of existing code **without changing its external behavior**.
*   **[⚡ Performance Optimizer (`performance-optimizer`)](v7.1/modes/worker/cross-functional/performance-optimizer/performance-optimizer.mode.md)**
    *   *Summary:* Identifies, analyzes, and resolves performance bottlenecks across the full stack using profiling, analysis, and optimization techniques. Measures impact against goals.
*   **[✍️ Technical Writer (`technical-writer`)](v7.1/modes/worker/cross-functional/technical-writer/technical-writer.mode.md)**
    *   *Summary:* Creates clear, accurate, and comprehensive documentation tailored to specific audiences, including READMEs, API documentation, user guides, and tutorials.
*   **[🌱 Junior Developer (`junior-developer`)](v7.1/modes/worker/cross-functional/junior-developer/junior-developer.mode.md)**
    *   *Summary:* Implements well-defined coding tasks based on clear requirements and guidance from senior team members. Focuses on writing clean code, learning project standards, creating basic unit tests, and contributing effectively to the team while developing skills.
*   **[🐛 Bug Fixer (`bug-fixer`)](v7.1/modes/worker/cross-functional/bug-fixer/bug-fixer.mode.md)**
    *   *Summary:* Systematically identifies, diagnoses, and resolves software bugs, implementing fixes and regression tests.
*   **[👀 Code Reviewer (`code-reviewer`)](v7.1/modes/worker/cross-functional/code-reviewer/code-reviewer.mode.md)**
    *   *Summary:* Reviews code changes for quality, standards adherence, bugs, security, performance, maintainability, and provides actionable feedback.
*   **[🔍 ESLint Specialist (`eslint-specialist`)](v7.1/modes/worker/cross-functional/eslint-specialist/eslint-specialist.mode.md)**
    *   *Summary:* A specialized tooling worker mode focused on implementing and managing ESLint configurations and rules.
*   **[🔧 Git Manager (`git-manager`)](v7.1/modes/worker/cross-functional/git-manager/git-manager.mode.md)**
    *   *Summary:* Executes Git commands (branch, merge, commit, push, pull, tag) safely, handles simple conflicts, and manages repository interactions.
*   **[🔧 Mode Maintainer (`mode-maintainer`)](v7.1/modes/worker/cross-functional/mode-maintainer/mode-maintainer.mode.md)**
    *   *Summary:* Applies specific, instructed modifications to existing mode definition files (.mode.md), ensuring structural integrity and adherence to templates.
*   **[🤔 Second Opinion (`second-opinion`)](v7.1/modes/worker/cross-functional/second-opinion/second-opinion.mode.md)**
    *   *Summary:* An independent, critical evaluator designed to rigorously assess proposed solutions, designs, code snippets, or approaches. It uses a structured evaluation framework considering correctness, efficiency, robustness, scalability, simplicity, standards compliance, and security.
*   **[🧑‍💻 Senior Developer (`senior-developer`)](v7.1/modes/worker/cross-functional/senior-developer/senior-developer.mode.md)**
    *   *Summary:* Designs, implements, and tests complex software components, ensuring code quality, maintainability, and adherence to best practices. Provides technical guidance and reviews code.
*   **[🧩 Complex Problem Solver (`complex-problem-solver`)](v7.1/modes/worker/cross-functional/complex-problem-solver/complex-problem-solver.mode.md)**
    *   *Summary:* Analyzes complex technical challenges, investigates root causes, evaluates solutions, and provides detailed recommendations for resolution.

### Domain: data-vis

*   **[📊 D3.js Specialist (`d3js-specialist`)](v7.1/modes/worker/data-vis/d3js-specialist/d3js-specialist.mode.md)**
    *   *Summary:* Specializes in creating dynamic, interactive data visualizations for the web using D3.js, focusing on best practices, accessibility, and performance.

### Domain: database

*   **[🍃 MongoDB Specialist (`mongodb-specialist`)](v7.1/modes/worker/database/mongodb-specialist/mongodb-specialist.mode.md)**
    *   *Summary:* Specializes in designing, implementing, querying, optimizing, and managing MongoDB databases, focusing on schema design, aggregation pipelines, indexing strategies, and performance optimization.
*   **[🐘 Neon DB Specialist (`neon-db-specialist`)](v7.1/modes/worker/database/neon-db-specialist/neon-db-specialist.mode.md)**
    *   *Summary:* Expert in designing, implementing, and managing Neon serverless PostgreSQL databases, including branching, connection pooling, and optimization.
*   **[🐬 MySQL Specialist (`mysql-specialist`)](v7.1/modes/worker/database/mysql-specialist/mysql-specialist.mode.md)**
    *   *Summary:* Designs schemas, writes/optimizes SQL queries, manages migrations, and ensures data integrity for MySQL databases.
*   **[💾 Database Specialist (`database-specialist`)](v7.1/modes/worker/database/database-specialist/database-specialist.mode.md)**
    *   *Summary:* Designs, implements, optimizes, and maintains SQL/NoSQL databases, focusing on schema design, ORMs, migrations, query optimization, data integrity, and performance.
*   **[🔄 dbt Specialist (`dbt-specialist`)](v7.1/modes/worker/database/dbt-specialist/dbt-specialist.mode.md)**
    *   *Summary:* A specialized data transformation mode focused on implementing and managing dbt projects. Expert in creating efficient data models, configuring transformations, and implementing testing strategies. Specializes in creating maintainable, well-documented data transformations that follow best practices for modern data warehouses.
*   **[🔍 Elasticsearch Specialist (`elasticsearch-specialist`)](v7.1/modes/worker/database/elasticsearch-specialist/elasticsearch-specialist.mode.md)**
    *   *Summary:* Expert in designing, implementing, querying, managing, and optimizing Elasticsearch clusters (across various versions) for diverse applications including full-text search, logging, analytics, and vector search.

### Domain: design

*   **[✨ One Shot Web Designer (`one-shot-web-designer`)](v7.1/modes/worker/design/one-shot-web-designer/one-shot-web-designer.mode.md)**
    *   *Summary:* Specializes in rapidly creating beautiful, creative web page visual designs (HTML/CSS/minimal JS) in a single session, focusing on aesthetic impact and delivering high-quality starting points.
*   **[🎨 UI Designer (`ui-designer`)](v7.1/modes/worker/design/ui-designer/ui-designer.mode.md)**
    *   *Summary:* Creates aesthetically pleasing and functional user interfaces, focusing on UX, visual design, wireframes, mockups, prototypes, and style guides while ensuring responsiveness and accessibility.
*   **[📊 Diagramer (`diagramer`)](v7.1/modes/worker/design/diagramer/diagramer.mode.md)**
    *   *Summary:* Translates conceptual descriptions into Mermaid syntax to create/update diagrams (graph, sequence, ER, C4, state, Gantt, etc.). Focuses on visualization, not analysis.

### Domain: devops

*   **[⚡ Cloudflare Workers Specialist (`cloudflare-workers-specialist`)](v7.1/modes/worker/devops/cloudflare-workers-specialist/cloudflare-workers-specialist.mode.md)**
    *   *Summary:* Specialized worker for developing and deploying Cloudflare Workers applications, including edge functions, service bindings (KV, R2, D1, Queues, DO, AI), asset management, Wrangler configuration, and performance optimization.
*   **[🏗️ Infrastructure Specialist (`infrastructure-specialist`)](v7.1/modes/worker/devops/infrastructure-specialist/infrastructure-specialist.mode.md)**
    *   *Summary:* Designs, implements, manages, and secures cloud/on-prem infrastructure using IaC (Terraform, CloudFormation, etc.), focusing on reliability, scalability, security, and cost.
*   **[🐳 Docker Compose Specialist (`docker-compose-specialist`)](v7.1/modes/worker/devops/docker-compose-specialist/docker-compose-specialist.mode.md)**
    *   *Summary:* Expert in designing, building, securing, and managing containerized applications with a focus on Docker Compose, Dockerfiles, and orchestration best practices.

### Domain: frontend

*   **[♿ Accessibility Specialist (`accessibility-specialist`)](v7.1/modes/worker/frontend/accessibility-specialist/accessibility-specialist.mode.md)**
    *   *Summary:* Audits UIs, implements fixes (HTML, CSS, ARIA), verifies WCAG compliance, generates reports, and guides teams on accessible design patterns.
*   **[⚛️ React Specialist (`react-specialist`)](v7.1/modes/worker/frontend/react-specialist/react-specialist.mode.md)**
    *   *Summary:* Specializes in building modern React applications using functional components, hooks, state management, performance optimization, and TypeScript integration.
*   **[⚡ Vite Specialist (`vite-specialist`)](v7.1/modes/worker/frontend/vite-specialist/vite-specialist.mode.md)**
    *   *Summary:* Expert in configuring, optimizing, and troubleshooting frontend tooling using Vite, including dev server, production builds, plugins, SSR, library mode, and migrations.
*   **[🎨 Material UI Specialist (`material-ui-specialist`)](v7.1/modes/worker/frontend/material-ui-specialist/material-ui-specialist.mode.md)**
    *   *Summary:* Implements UIs using the Material UI (MUI) ecosystem (Core, Joy, Base) for React, focusing on components, theming, styling (`sx`, `styled`), and Material Design principles.
*   **[🎯 jQuery Specialist (`jquery-specialist`)](v7.1/modes/worker/frontend/jquery-specialist/jquery-specialist.mode.md)**
    *   *Summary:* Specializes in implementing and managing jQuery-based applications, focusing on efficient DOM manipulations, handling events, AJAX calls, plugin integration, and managing jQuery modules, while adhering to modern JavaScript practices where applicable.
*   **[💚 Vue.js Developer (`vuejs-developer`)](v7.1/modes/worker/frontend/vuejs-developer/vuejs-developer.mode.md)**
    *   *Summary:* Expertly builds modern, performant UIs and SPAs using Vue.js (v2/v3), Composition API, Options API, Vue Router, and Pinia/Vuex.
*   **[💨 Tailwind CSS Specialist (`tailwind-specialist`)](v7.1/modes/worker/frontend/tailwind-specialist/tailwind-specialist.mode.md)**
    *   *Summary:* Implements modern, responsive UIs using Tailwind CSS, with expertise in utility classes, configuration customization, responsive design, and optimization for production.
*   **[💿 Remix Developer (`remix-developer`)](v7.1/modes/worker/frontend/remix-developer/remix-developer.mode.md)**
    *   *Summary:* Expert in developing fast, resilient, full-stack web applications using Remix, focusing on routing, data flow, progressive enhancement, and server/client code colocation.
*   **[🔑 Clerk Auth Specialist (`clerk-auth-specialist`)](v7.1/modes/worker/frontend/clerk-auth-specialist/clerk-auth-specialist.mode.md)**
    *   *Summary:* Specializes in implementing secure authentication and user management using Clerk, covering frontend/backend integration, route protection, session handling, and advanced features.
*   **[🔥 SvelteKit Developer (`sveltekit-developer`)](v7.1/modes/worker/frontend/sveltekit-developer/sveltekit-developer.mode.md)**
    *   *Summary:* Specializes in building high-performance web applications using the SvelteKit framework, covering routing, data loading, form handling, SSR/SSG, and deployment.
*   **[🔷 TypeScript Specialist (`typescript-specialist`)](v7.1/modes/worker/frontend/typescript-specialist/typescript-specialist.mode.md)**
    *   *Summary:* Specializes in writing, configuring, and improving strongly-typed JavaScript applications using TypeScript.
*   **[🖥️ Frontend Developer (`frontend-developer`)](v7.1/modes/worker/frontend/frontend-developer/frontend-developer.mode.md)**
    *   *Summary:* Generalist for foundational UI development (HTML, CSS, Vanilla JS), basic interactivity, API integration, and coordinating/delegating to frontend specialists.
*   **[🧊 Three.js Specialist (`threejs-specialist`)](v7.1/modes/worker/frontend/threejs-specialist/threejs-specialist.mode.md)**
    *   *Summary:* Specializes in creating 3D graphics and animations for the web using Three.js, including scene setup, materials, lighting, models (glTF), shaders (GLSL), and performance optimization.
*   **[🧑‍🚀 Astro Developer (`astro-developer`)](v7.1/modes/worker/frontend/astro-developer/astro-developer.mode.md)**
    *   *Summary:* Specializes in building fast, content-focused websites and applications with the Astro framework, focusing on island architecture, content collections, integrations, performance, SSR, and Astro DB/Actions.
*   **[🧩 Shadcn UI Specialist (`shadcn-ui-specialist`)](v7.1/modes/worker/frontend/shadcn-ui-specialist/shadcn-ui-specialist.mode.md)**
    *   *Summary:* Specializes in building UIs using Shadcn UI components with React and Tailwind CSS, focusing on composition, customization via CLI, and accessibility.
*   **[🚀 Next.js Developer (`nextjs-developer`)](v7.1/modes/worker/frontend/nextjs-developer/nextjs-developer.mode.md)**
    *   *Summary:* Expert in building efficient, scalable full-stack web applications using Next.js, specializing in App Router, Server/Client Components, advanced data fetching, Server Actions, rendering strategies, API routes, Vercel deployment, and performance optimization.
*   **[🅱️ Bootstrap Specialist (`bootstrap-specialist`)](v7.1/modes/worker/frontend/bootstrap-specialist/bootstrap-specialist.mode.md)**
    *   *Summary:* Specializes in building responsive websites and applications using the Bootstrap framework (v4 & v5), focusing on grid mastery, component usage, utilities, customization, and accessibility.

#### Sub-Domain: angular

*   **[🅰️ Angular Developer (`angular-developer`)](v7.1/modes/worker/frontend/angular/angular-developer/angular-developer.mode.md)**
    *   *Summary:* Expert in developing robust, scalable, and maintainable Angular applications using TypeScript, with a focus on best practices, performance, testing, and integration with Angular ecosystem tools.

#### Sub-Domain: animation

*   **[✨ anime.js Specialist (`animejs-specialist`)](v7.1/modes/worker/frontend/animation/animejs-specialist/animejs-specialist.mode.md)**
    *   *Summary:* Expert in creating complex, performant web animations using anime.js, including timelines, SVG morphing, interactive, and scroll-triggered effects.

#### Sub-Domain: ui-library

*   **[🐜 Ant Design Specialist (`ant-design-specialist`)](v7.1/modes/worker/frontend/ui-library/ant-design-specialist/ant-design-specialist.mode.md)**
    *   *Summary:* Implements and customizes React components using Ant Design, focusing on responsiveness, accessibility, performance, and best practices.

### Domain: qa

*   **[🎭 E2E Testing Specialist (`e2e-tester`)](v7.1/modes/worker/qa/e2e-tester/e2e-tester.mode.md)**
    *   *Summary:* Designs, writes, executes, and maintains End-to-End (E2E) tests using frameworks like Cypress, Playwright, Selenium to simulate user journeys and ensure application quality.
*   **[🔗 Integration Tester (`integration-tester`)](v7.1/modes/worker/qa/integration-tester/integration-tester.mode.md)**
    *   *Summary:* Verifies interactions between components, services, or systems, focusing on interfaces, data flow, and contracts using API testing, mocks, and stubs.

