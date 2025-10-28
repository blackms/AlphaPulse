# ADR 005: Billing and Subscription Management System

Date: 2025-10-20
Status: Proposed

## Context

As AlphaPulse transitions to a multi-tenant SaaS model, we need a billing and subscription management system that:

- **Supports tiered pricing**: Starter ($99/mo), Pro ($499/mo), Enterprise (custom)
- **Enables usage-based billing**: API calls, trades executed, data storage, compute hours
- **Handles subscription lifecycle**: Trial period, upgrades/downgrades, cancellations, refunds
- **Integrates with payment processors**: Credit cards, ACH, wire transfers (for Enterprise)
- **Provides invoicing**: Automatic invoice generation, tax calculation, payment reminders
- **Supports compliance**: PCI-DSS, GDPR, SOX for financial data
- **Offers self-service**: Customer portal for plan changes, invoice downloads, payment method updates
- **Enables revenue analytics**: MRR, churn rate, LTV, cohort analysis

### Pricing Model Recap

| Tier | Monthly Price | Key Limits | Payment Method |
|------|---------------|------------|----------------|
| **Starter** | $99 | 3 agents, 5 positions, 10k API calls/day | Credit card only |
| **Pro** | $499 | All 6 agents, 25 positions, 100k API calls/day | Credit card, ACH |
| **Enterprise** | Custom (starts at $2,000) | Unlimited, dedicated resources | All methods + invoicing |

### Usage-Based Add-Ons

- **Additional API calls**: $10 per 10k calls
- **Additional positions**: $5 per position per month
- **Historical data access**: $50/month for 5-year history (vs 30 days free for Starter)
- **Priority support**: $200/month (included in Enterprise)

### Billing Requirements

1. **Subscription Management**:
   - Automatic recurring billing (monthly)
   - 14-day free trial (credit card required)
   - Proration on plan changes (upgrade mid-month → charge difference)
   - Grace period on payment failure (7 days before suspension)

2. **Usage Tracking & Metering**:
   - Track API calls, trades, data usage per tenant
   - Aggregate usage metrics daily
   - Generate overage charges at end of billing cycle
   - Real-time usage dashboards for tenants

3. **Payment Processing**:
   - Support major credit cards (Visa, Mastercard, Amex)
   - ACH/bank transfers for Pro/Enterprise
   - International payments (multi-currency support)
   - PCI-DSS compliance (never store card details directly)

4. **Invoicing & Receipts**:
   - Automatic invoice generation on billing date
   - PDF invoices with company branding
   - Tax calculation (US sales tax, EU VAT)
   - Email delivery + customer portal download

5. **Revenue Recognition**:
   - Deferred revenue accounting (recognize monthly over subscription period)
   - Integration with accounting systems (QuickBooks, Xero)
   - Revenue reports for financial statements

6. **Customer Self-Service**:
   - Update payment method
   - Change subscription plan (upgrade/downgrade)
   - View invoice history
   - Download receipts
   - Cancel subscription (with exit survey)

## Decision

We will use **Stripe Billing** as our primary billing platform with custom usage metering:

### Architecture: Stripe Billing + Internal Usage Tracking

**Components:**

1. **Stripe Billing** (External SaaS)
   - Subscription management (plans, trials, upgrades/downgrades)
   - Payment processing (cards, ACH, international)
   - Invoicing (automatic generation, tax calculation)
   - Customer portal (self-service plan changes)
   - Webhook events (payment success, subscription cancelled, invoice created)

2. **Usage Metering Service** (Internal - New Microservice)
   - Track API calls, trades, positions per tenant
   - Aggregate usage data hourly → daily → monthly
   - Report usage to Stripe via Usage Records API
   - Database schema:
     ```sql
     CREATE TABLE usage_events (
       id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
       tenant_id UUID NOT NULL,
       event_type VARCHAR(50) NOT NULL, -- 'api_call', 'trade', 'position_day'
       quantity INT NOT NULL DEFAULT 1,
       timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
       metadata JSONB
     );

     CREATE TABLE usage_aggregates (
       tenant_id UUID NOT NULL,
       metric VARCHAR(50) NOT NULL, -- 'api_calls', 'trades', 'position_days'
       period_start DATE NOT NULL,
       period_end DATE NOT NULL,
       quantity INT NOT NULL,
       reported_to_stripe BOOLEAN DEFAULT FALSE,
       PRIMARY KEY (tenant_id, metric, period_start)
     );
     ```

3. **Billing Middleware** (Application Layer)
   - Intercept API requests → increment usage counters
   - Check quota limits before processing request
   - Reject requests if quota exceeded (with upgrade prompt)
   - Example:
     ```python
     @app.middleware("http")
     async def billing_middleware(request: Request, call_next):
         tenant_id = extract_tenant_from_jwt(request)

         # Increment API call counter
         await usage_service.record_event(
             tenant_id=tenant_id,
             event_type='api_call',
             quantity=1
         )

         # Check quota
         usage = await usage_service.get_current_usage(tenant_id, 'api_calls')
         quota = await billing_service.get_quota(tenant_id, 'api_calls')

         if usage >= quota:
             return JSONResponse(
                 status_code=429,
                 content={"error": "API quota exceeded", "upgrade_url": "/billing/upgrade"}
             )

         response = await call_next(request)
         return response
     ```

4. **Stripe Webhook Handler** (Event Processing)
   - Listen for Stripe events: `customer.subscription.created`, `invoice.payment_succeeded`, `customer.subscription.deleted`
   - Update tenant status in database (active, suspended, cancelled)
   - Send email notifications (payment success, payment failed, cancellation)
   - Example workflow:
     ```python
     @app.post("/webhooks/stripe")
     async def handle_stripe_webhook(request: Request):
         event = stripe.Webhook.construct_event(
             payload=await request.body(),
             sig_header=request.headers.get('stripe-signature'),
             secret=STRIPE_WEBHOOK_SECRET
         )

         if event.type == 'invoice.payment_failed':
             tenant_id = event.data.object.metadata.tenant_id
             await tenant_service.suspend(tenant_id, reason='payment_failed')
             await notification_service.send_payment_failed_email(tenant_id)

         elif event.type == 'customer.subscription.deleted':
             tenant_id = event.data.object.metadata.tenant_id
             await tenant_service.cancel(tenant_id)
             await notification_service.send_cancellation_confirmed(tenant_id)

         return {"status": "ok"}
     ```

5. **Admin Billing Dashboard** (Internal Portal)
   - View all tenants with subscription status
   - Manually adjust billing (refunds, credits, discounts)
   - Override quotas for special cases
   - Revenue analytics (MRR, churn, LTV)

### Stripe Integration Details

**Stripe Products & Prices:**

```python
# Create products in Stripe (one-time setup)
starter_product = stripe.Product.create(
    name="AlphaPulse Starter",
    description="3 agents, 5 positions, 10k API calls/day"
)

starter_price = stripe.Price.create(
    product=starter_product.id,
    unit_amount=9900,  # $99.00 in cents
    currency="usd",
    recurring={"interval": "month"}
)

# Usage-based pricing for overage
api_calls_price = stripe.Price.create(
    product="api_calls_addon",
    unit_amount=100,  # $1.00 per 100 calls (billed at end of month)
    currency="usd",
    recurring={"interval": "month", "usage_type": "metered"}
)
```

**Subscription Creation:**

```python
# When tenant signs up
customer = stripe.Customer.create(
    email=tenant_email,
    metadata={"tenant_id": str(tenant_id)}
)

subscription = stripe.Subscription.create(
    customer=customer.id,
    items=[
        {"price": starter_price.id},  # Base subscription
        {"price": api_calls_price.id}  # Metered usage
    ],
    trial_period_days=14,
    metadata={"tenant_id": str(tenant_id)}
)
```

**Usage Reporting:**

```python
# Daily job: Report usage to Stripe
async def report_usage_to_stripe(tenant_id: UUID, date: date):
    # Get aggregated usage for the day
    usage = await db.fetch_one(
        "SELECT quantity FROM usage_aggregates WHERE tenant_id = $1 AND period_start = $2",
        tenant_id, date
    )

    # Get Stripe subscription item ID
    subscription_item_id = await get_stripe_subscription_item(tenant_id, 'api_calls')

    # Report to Stripe
    stripe.SubscriptionItem.create_usage_record(
        subscription_item_id,
        quantity=usage.quantity,
        timestamp=int(date.timestamp()),
        action='set'  # Overwrite previous report (idempotent)
    )

    # Mark as reported
    await db.execute(
        "UPDATE usage_aggregates SET reported_to_stripe = TRUE WHERE tenant_id = $1 AND period_start = $2",
        tenant_id, date
    )
```

### Payment Flow

1. **Tenant Signs Up**:
   - Fill out registration form (email, company, plan)
   - Enter credit card details (Stripe Checkout or Elements)
   - Stripe creates customer + subscription (with 14-day trial)
   - Webhook triggers tenant provisioning (ADR-002)

2. **Trial Period (Days 1-14)**:
   - Full access to all features
   - No charges
   - Usage tracked but not billed
   - Reminder email on day 10: "Trial ending in 4 days"

3. **First Billing (Day 15)**:
   - Stripe charges card for monthly subscription ($99)
   - Invoice generated and emailed
   - Subscription status: active
   - Usage meter resets for new billing cycle

4. **Ongoing Billing (Monthly)**:
   - Charge base subscription fee
   - Add overage charges (if any): `api_calls_used - api_calls_included) * price_per_call`
   - Generate invoice
   - Email receipt
   - If payment fails → retry 3 times over 7 days → suspend account

5. **Plan Upgrade (Mid-Month)**:
   - Tenant clicks "Upgrade to Pro"
   - Stripe calculates proration: `(days_remaining / days_in_month) * (new_price - old_price)`
   - Charge prorated difference immediately
   - Update subscription
   - Increase quotas in application

6. **Cancellation**:
   - Tenant clicks "Cancel Subscription"
   - Show exit survey (optional feedback)
   - Subscription remains active until end of billing period
   - After billing period ends → status = cancelled → restrict access (read-only for 30 days)

### Revenue Analytics

**Key Metrics (tracked in internal database + Stripe Reporting API):**

1. **Monthly Recurring Revenue (MRR)**:
   ```sql
   SELECT SUM(amount / 100) AS mrr
   FROM stripe_subscriptions
   WHERE status = 'active';
   ```

2. **Churn Rate**:
   ```sql
   SELECT COUNT(*) FILTER (WHERE cancelled_at >= date_trunc('month', NOW()))
          / COUNT(*) FILTER (WHERE created_at < date_trunc('month', NOW()))
   AS churn_rate
   FROM stripe_subscriptions;
   ```

3. **Customer Lifetime Value (LTV)**:
   ```sql
   SELECT AVG(total_revenue) AS ltv
   FROM (
     SELECT customer_id, SUM(amount_paid / 100) AS total_revenue
     FROM stripe_invoices
     WHERE status = 'paid'
     GROUP BY customer_id
   ) AS customer_revenues;
   ```

4. **Average Revenue Per User (ARPU)**:
   ```sql
   SELECT SUM(amount_paid) / COUNT(DISTINCT customer_id) AS arpu
   FROM stripe_invoices
   WHERE period_start >= date_trunc('month', NOW());
   ```

**Dashboard**: Build custom Grafana/Metabase dashboard with these metrics (updated daily).

### Tax Compliance

**Stripe Tax** (add-on service):
- Automatically calculates sales tax (US) and VAT (EU) based on customer location
- Handles tax registration and filing
- Cost: 0.5% of transaction volume (worth it to avoid manual tax management)

**Alternative**: TaxJar API integration (if Stripe Tax too expensive).

## Consequences

### Positive

✅ **Industry Leader**: Stripe is battle-tested by 1M+ businesses (including OpenAI, Shopify, Salesforce)
✅ **Feature Complete**: Supports all requirements (subscriptions, usage billing, invoicing, tax)
✅ **Developer Experience**: Excellent API documentation, client libraries (Python), webhooks
✅ **PCI Compliance**: Stripe handles all PCI-DSS requirements (we never store card data)
✅ **Customer Portal**: Built-in self-service portal (save 2+ weeks of development time)
✅ **International**: Supports 135+ currencies, 45+ countries
✅ **Revenue Analytics**: Stripe Sigma (SQL queries on billing data) for custom reports
✅ **Integration Ecosystem**: Pre-built integrations with QuickBooks, Xero, NetSuite
✅ **Scalability**: Handles billions of dollars in transactions (no scaling concerns)

### Negative

⚠️ **Cost**: 2.9% + $0.30 per transaction (e.g., $99 charge → $3.17 fee = 3.2%)
⚠️ **Vendor Lock-In**: Difficult to migrate to another provider (customer data, payment methods in Stripe)
⚠️ **Complex Pricing**: Usage-based billing requires custom metering infrastructure
⚠️ **Webhook Reliability**: Must handle webhook retries and idempotency correctly
⚠️ **Tax Add-On Cost**: Stripe Tax adds 0.5% of transaction volume (~$500/month at $100k MRR)

### Mitigation Strategies

1. **Cost Optimization**: Negotiate volume discount with Stripe (after $100k/month revenue)
2. **Abstraction Layer**: Build `BillingService` interface to abstract Stripe (easier to migrate later)
3. **Webhook Resilience**: Use idempotency keys, retry logic, dead-letter queue for failed events
4. **Tax Strategy**: Start with manual tax handling, enable Stripe Tax at $50k+ MRR
5. **Monitoring**: Alert on webhook processing failures, payment decline spikes, unusual churn

## Alternatives Considered

### Option A: Chargebee (Subscription Management Platform)

**Pros:**
- More flexible subscription management (trials, coupons, dunning)
- Better support for complex B2B pricing (negotiated contracts)
- Unified billing for multiple payment processors (Stripe, PayPal, Authorize.net)
- Better revenue recognition features

**Cons:**
- ❌ **Additional cost**: $300/month base fee + transaction fees (vs Stripe: pay-as-you-go)
- ❌ **Complexity**: Another vendor to integrate with (Chargebee + Stripe)
- ❌ **Overkill**: Advanced features not needed for MVP (3 tiers, simple pricing)

**Why Rejected**: Too expensive and complex for initial launch. Revisit at 500+ customers if we need advanced features (e.g., complex contract management for Enterprise).

### Option B: Build Custom Billing System

**Pros:**
- Full control over features and pricing
- No transaction fees (just payment processor fees)
- Optimized for our specific use case

**Cons:**
- ❌ **Development cost**: 6+ months to build subscription management, invoicing, tax handling
- ❌ **PCI compliance**: Expensive and time-consuming to achieve certification
- ❌ **Maintenance burden**: Ongoing development for new payment methods, regulations
- ❌ **Opportunity cost**: Engineering time better spent on core product features

**Why Rejected**: Not core competency. Billing is complex (tax, compliance, payment methods) and risky to build in-house. Stripe provides more reliability than we could achieve in 6 months.

### Option C: Paddle (Merchant of Record)

**Pros:**
- Simplest setup: Paddle handles all payment processing, tax, compliance
- No PCI compliance burden
- Global tax handling included (VAT, sales tax)
- Single 1099 for revenue (simplified accounting)

**Cons:**
- ❌ **Higher fees**: 5% + $0.50 per transaction (vs Stripe: 2.9% + $0.30)
- ❌ **Less control**: Paddle owns customer relationship (email, billing)
- ❌ **Limited customization**: Checkout flow less flexible
- ❌ **Slower payouts**: Monthly payouts (vs Stripe: 2-day rolling)

**Why Rejected**: Higher fees significantly impact margins (5% vs 3% = $2,000/month at $100k MRR). Loss of customer relationship is dealbreaker for B2B SaaS (we need direct communication with tenants).

### Option D: Open-Source Billing (Kill Bill)

**Pros:**
- Free and open-source
- Highly customizable
- No transaction fees
- No vendor lock-in

**Cons:**
- ❌ **Complexity**: Steep learning curve, Java-based (our stack is Python)
- ❌ **Self-hosted**: Requires dedicated infrastructure and maintenance
- ❌ **No payment processing**: Still need to integrate Stripe/Braintree for payments
- ❌ **Limited community**: Smaller community than Stripe (harder to find help)

**Why Rejected**: Too complex for MVP. Requires significant upfront investment in infrastructure and learning. Stripe provides faster time-to-market.

## Implementation Plan

### Phase 1: Stripe Setup (Sprint 1)

1. Create Stripe account (Production + Test mode)
2. Define products and prices in Stripe dashboard
3. Configure webhook endpoints
4. Set up Stripe Tax (or TaxJar integration)
5. Test checkout flow in Stripe sandbox

### Phase 2: Usage Metering Service (Sprint 2)

1. Create database schema for usage tracking
2. Implement `UsageMeteringService` class
3. Add billing middleware to FastAPI (intercept requests, track usage)
4. Daily background job to aggregate usage and report to Stripe
5. Unit tests + integration tests

### Phase 3: Subscription Management API (Sprint 2-3)

1. Create `BillingService` abstraction layer
2. Implement subscription CRUD operations (create, update, cancel)
3. REST endpoints: `POST /subscriptions`, `GET /subscriptions/{id}`, `PATCH /subscriptions/{id}`, `DELETE /subscriptions/{id}`
4. Integrate with Stripe API
5. Error handling for payment failures

### Phase 4: Webhook Processing (Sprint 3)

1. Implement webhook handler endpoint (`POST /webhooks/stripe`)
2. Verify webhook signatures (security)
3. Handle all relevant events (payment success, failure, subscription cancelled, trial ending)
4. Update tenant status in database
5. Send email notifications
6. Idempotency checks (process each event exactly once)

### Phase 5: Customer Portal (Sprint 4)

1. Integrate Stripe Customer Portal (pre-built UI)
2. Add "Manage Billing" button in dashboard → redirect to Stripe portal
3. Configure portal settings (allow plan changes, payment method updates, cancellation)
4. Test self-service flows

### Phase 6: Admin Dashboard (Sprint 4)

1. Build internal admin UI for billing management
2. View all subscriptions (status, plan, MRR)
3. Manually issue refunds, credits, discounts
4. Revenue analytics charts (MRR, churn, LTV)
5. Export invoice data to CSV

### Phase 7: Testing & Launch (Sprint 5)

1. End-to-end testing with test credit cards
2. Simulate payment failures, subscription cancellations
3. Load testing (webhook processing at scale)
4. Security audit (PCI-DSS, GDPR compliance)
5. Documentation: "How to Upgrade Your Plan", "Payment Failure FAQs"

## Links

- Issue: [To be created - Billing System Epic]
- Related: ADR-002 (Tenant Provisioning), ADR-001 (Data Isolation)
- Reference: [Stripe Billing Documentation](https://stripe.com/docs/billing)
- Reference: [Stripe Usage-Based Billing](https://stripe.com/docs/billing/subscriptions/usage-based)
- Reference: [SaaS Billing Best Practices](https://www.paddle.com/resources/saas-billing)
