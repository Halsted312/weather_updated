# Plan Management Instructions

## Plan Lifecycle

1. **Active Plans**: Store in `.claude/plans/active/`
2. **Completed Plans**: Move to `.claude/plans/completed/` with completion date prefix
3. **Naming**: `{slug-name}.md` (e.g., `vc-schema-greenfield.md`)

## Plan Structure Requirements

Every plan MUST include:

### Header Block
```yaml
---
plan_id: toasty-roaming-boole
created: 2025-01-15
status: in_progress  # draft | in_progress | blocked | completed
priority: high       # low | medium | high | critical
agent: kalshi-weather-quant
---
```

### Required Sections

1. **Objective** - One-sentence goal
2. **Context** - Why this matters, links to prior work
3. **Tasks** - Checkbox list with clear deliverables
4. **Files to Create/Modify** - Explicit paths
5. **Completion Criteria** - How we know it's done
6. **Sign-off Log** - Date-stamped progress notes

## Sign-off Protocol

When pausing work on a plan, append to the Sign-off Log:
```markdown
## Sign-off Log

### 2025-01-15 14:30 CST
**Status**: In progress - 40% complete
**Last completed**: 
- ✅ Created `src/config/vc_elements.py`
- ✅ Added ORM models to `src/db/models.py`

**Next steps**:
1. Create Alembic migration with constraints
2. Seed `wx.vc_location` table

**Blockers**: None

**Context for next session**: 
Models are added but CHECK/UNIQUE constraints will be in migration, not ORM level.
```

## Resuming Work

When resuming a plan:
1. Read the full plan file
2. Check the Sign-off Log for last state
3. Verify file states match expected
4. Continue from "Next steps"