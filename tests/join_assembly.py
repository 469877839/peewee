"""
Independent-reference tests for joined-object assembly.

The fixture rows are generated programmatically from a fixed seed and cover:

* the same model appearing multiple times in one query (self-join aliases),
* consecutive LEFT JOINs (chains with missing and all-NULL middle links),
* foreign keys that do not point at a primary key (a unique column, and a
  unique column on a composite-primary-key model),
* selected columns that are not grouped by model (interleaved order),
* rows that exist but have NULL in every non-key column, kept distinct from
  rows that do not exist at all (LEFT JOIN miss).

Expected object graphs are derived from raw rows read straight out of
SQLite plus an explicit declaration of each join path; the assembler under
test is never consulted for the reference values.  Assertions compare
entity values, association attribution and None for misses -- never SQL
strings.
"""
import random

from peewee import *

from .base import ModelTestCase
from .base_models import TestModel


# Fixed seed for the generated fixture data.
SEED = 20261111


class JAEmployee(TestModel):
    name = CharField(null=True)
    code = CharField(null=True, unique=True)
    manager = ForeignKeyField('self', null=True)


class JADepartment(TestModel):
    title = CharField(null=True)
    # Foreign key to a unique non-primary-key column.
    head = ForeignKeyField(JAEmployee, field='code', null=True)


class JATicket(TestModel):
    note = CharField(null=True)
    department = ForeignKeyField(JADepartment, null=True)
    assignee = ForeignKeyField(JAEmployee, null=True)


class JAPair(TestModel):
    left = IntegerField()
    right = IntegerField()
    label = CharField(null=True, unique=True)
    class Meta:
        # Composite primary key; JASlot.pair points at the non-pk label.
        primary_key = CompositeKey('left', 'right')


class JASlot(TestModel):
    tag = CharField(null=True)
    pair = ForeignKeyField(JAPair, field='label', null=True)


MODELS = [JAEmployee, JADepartment, JATicket, JAPair, JASlot]


def load_world(database, models):
    """Read every fixture table back as raw rows, keyed by primary key."""
    world = {}
    for model in models:
        fields = model._meta.sorted_fields
        pk_fields = model._meta.get_primary_keys()
        sql = 'SELECT %s FROM "%s"' % (
            ', '.join('"%s"' % f.column_name for f in fields),
            model._meta.table_name)
        cursor = database.execute_sql(sql)
        rows = {}
        for values in cursor.fetchall():
            row = dict(zip([f.column_name for f in fields], values))
            key = tuple(row[f.column_name] for f in pk_fields)
            rows[key] = row
        world[model] = rows
    return world


class Relation(object):
    """One declared hop of a join path, expressed in raw-table terms.

    ``src`` is the model of the node the hop starts from (used to validate
    the declaration while walking); ``fk_col`` is the column on the source
    row holding the reference value; ``dest_key`` is the column on the
    destination row it must match; ``attr`` is the attribute the assembler
    should attach the destination object to; ``dest_model`` is the model
    class of the destination role (aliases still instantiate their
    underlying model).
    """
    def __init__(self, src, fk_col, dest_model, dest_key, attr):
        self.src = src
        self.fk_col = fk_col
        self.dest_model = dest_model
        self.dest_key = dest_key
        self.attr = attr


def resolve_path(world, root_model, root_key, path):
    """Walk ``path`` over raw rows; None marks a verified-absent node."""
    nodes = [(root_model, world[root_model][root_key])]
    for rel in path:
        parent_model, parent = nodes[-1]
        if parent_model is not rel.src:
            raise ValueError('path declaration expects %r, got %r' %
                             (rel.src, parent_model))
        if parent is None:
            nodes.append((rel.dest_model, None))
            continue
        fk_value = parent[rel.fk_col]
        node = None
        if fk_value is not None:
            for row in world[rel.dest_model].values():
                if row[rel.dest_key] == fk_value:
                    node = row
                    break
        nodes.append((rel.dest_model, node))
    return nodes


def walk_orm(root_obj, path):
    """Follow the same path on the assembled ORM objects."""
    nodes = [(type(root_obj), root_obj)]
    for rel in path:
        parent = nodes[-1][1]
        nodes.append((rel.dest_model,
                      None if parent is None else getattr(parent, rel.attr)))
    return nodes


def obj_pk_key(model, obj):
    return tuple(obj.__data__[f.name]
                 for f in model._meta.get_primary_keys())


class JoinAssemblyTestCase(ModelTestCase):
    requires = MODELS

    def setUp(self):
        super(JoinAssemblyTestCase, self).setUp()
        self.generate_fixtures()
        self.world = load_world(self.database, MODELS)

    def generate_fixtures(self):
        raise NotImplementedError

    def root_keys(self, model):
        """Raw primary keys of the root table, in id order."""
        pk_fields = model._meta.get_primary_keys()
        sql = 'SELECT %s FROM "%s" ORDER BY %s' % (
            ', '.join('"%s"' % f.column_name for f in pk_fields),
            model._meta.table_name,
            ', '.join('"%s"' % f.column_name for f in pk_fields))
        return [tuple(r) for r in self.database.execute_sql(sql).fetchall()]

    def assertEntity(self, obj, row, model, fields=None):
        """Every selected field on ``obj`` matches the raw ``row`` values."""
        self.assertIsNotNone(obj)
        self.assertIsInstance(obj, model)
        if fields is None:
            fields = model._meta.sorted_fields
        names = [getattr(field, 'name', field) for field in fields]
        for name in names:
            column = model._meta.fields[name].column_name
            self.assertEqual(
                obj.__data__[name], row[column],
                '%s.%s: orm=%r raw=%r' % (model.__name__, name,
                                          obj.__data__[name], row[column]))

    def assertGraph(self, orm_roots, root_model, paths, raw_keys):
        """Compare assembled graphs against the raw-row reference.

        Each entry of ``paths`` is a list of Relation hops starting at the
        root; the root alone is an empty path.  ``raw_keys`` must be read
        with root_keys() before entering a query-count block, so that the
        only query issued inside the block is the ORM query under test --
        walking the assembled graph afterwards may not lazy-load anything.
        """
        self.assertEqual(
            [obj_pk_key(root_model, obj) for obj in orm_roots],
            raw_keys)
        for obj, root_key in zip(orm_roots, raw_keys):
            for path in paths:
                expected = resolve_path(self.world, root_model,
                                        root_key, path)
                actual = walk_orm(obj, path)
                self.assertEqual(len(actual), len(expected))
                for (model, node_obj), (_, node_row) in zip(actual, expected):
                    if node_row is None:
                        self.assertIsNone(node_obj)
                    else:
                        self.assertEntity(node_obj, node_row, model)



class TestJoinAssembly(JoinAssemblyTestCase):
    """Generated fixtures plus independent-reference assembly checks.

    The generator is deterministic (fixed seed) and deliberately seeds every
    edge category instead of hoping the rng reaches them:

    * id 4 employee, id 3 department, id 5 ticket and id 2 slot exist with
      NULL in every non-key column;
    * NULL fks, dangling fks (no matching row) and matched fks all occur;
    * every join chain contains at least one full hit including a hit whose
      intermediate node is an all-NULL row;
    * grandparent/manager-of-manager rows exercise three appearances of the
      same model in one query.
    """

    def generate_fixtures(self):
        rng = random.Random(SEED)
        # Free-text filler still goes through the seeded rng; the structural
        # links below are explicit so coverage does not depend on rng luck.
        suffixes = [rng.choice(['x', 'y', 'z']) for _ in range(10)]

        # (id, name, code, manager_id).  id=4 is all-null non-key.
        emp_rows = [
            (1, 'ada', 'E1', None),       # top manager
            (2, 'bob', 'E2', 1),          # manager chain
            (3, 'cy', 'E3', 2),           # chain reaches a grandparent
            (4, None, None, None),        # exists, all non-key NULL
            (5, 'eve', 'E5', 4),          # manager is the all-null row
            (6, 'fay', 'E6', None),
            (7, 'gus', 'E7', None),
            (8, 'hal%s' % suffixes[0], 'E8', 9901),  # dangling manager
        ]
        JAEmployee.insert_many(emp_rows, fields=[
            JAEmployee.id, JAEmployee.name, JAEmployee.code,
            JAEmployee.manager]).execute()

        # (id, title, head_code): fk targets the non-pk unique code.
        dept_rows = [
            (1, 'dept-1', 'E1'),          # head found
            (2, 'dept-2%s' % suffixes[1], None),  # no head (null fk)
            (3, None, None),              # exists, all non-key NULL
            (4, 'dept-4', 'E999'),        # dangling head code
            (5, 'dept-5', 'E3'),          # head found
        ]
        JADepartment.insert_many(dept_rows, fields=[
            JADepartment.id, JADepartment.title,
            JADepartment.head]).execute()

        # (id, note, department_id, assignee_id).  id=5 all-null non-key.
        ticket_rows = [
            (1, 'note-1', 1, 2),          # chain fully hits
            (2, 'note-2', 5, 8),          # dept hits, head hits (E3)
            (3, 'note-3', 3, 5),          # dept is all-null row, head null
            (4, 'note-4', 4, 4),          # dept hits, head dangles
            (5, None, None, None),        # exists, all non-key NULL
            (6, 'note-6', None, None),    # null fks
            (7, 'note-7%s' % suffixes[2], 9903, 9902),  # both dangle
            (8, 'note-8', 2, 3),          # chain hits; assignee hits
        ]
        JATicket.insert_many(ticket_rows, fields=[
            JATicket.id, JATicket.note, JATicket.department,
            JATicket.assignee]).execute()

        # Composite-pk table; label is the non-pk unique fk target.
        pair_rows = [
            (1, 10, 'P1'),
            (2, 20, 'P2'),
            (3, 30, None),                # exists, non-key label NULL
        ]
        JAPair.insert_many(pair_rows, fields=[
            JAPair.left, JAPair.right, JAPair.label]).execute()
        slot_rows = [
            (1, 's1', 'P1'),              # pair found
            (2, None, None),              # exists, all non-key NULL
            (3, 's3%s' % suffixes[3], 'P2'),  # pair found
            (4, 's4', 'P999'),            # dangling label
        ]
        JASlot.insert_many(slot_rows, fields=[
            JASlot.id, JASlot.tag, JASlot.pair]).execute()

    # ------------------------------------------------------------------
    # coverage declarations, checked against the raw rows rather than
    # against anything the assembler produced.
    # ------------------------------------------------------------------
    def test_fixture_coverage(self):
        categories = self._relation_categories()
        # Every declared relation sees each possible outcome at least once.
        for label, counts in categories.items():
            for outcome in ('hit', 'null-fk', 'dangling'):
                self.assertGreater(
                    counts[outcome], 0,
                    'relation %r lacks %r coverage: %r' %
                    (label, outcome, counts))
        # The two chained relations also see parent-miss propagation.
        self.assertGreater(categories['ticket->dept']['parent-miss'], 0)
        self.assertGreater(categories['emp->mgr']['parent-miss'], 0)

        # Existing rows whose non-key columns are all NULL must be present.
        all_null_emp = self._all_null_rows(JAEmployee)
        all_null_dept = self._all_null_rows(JADepartment)
        all_null_ticket = self._all_null_rows(JATicket)
        all_null_slot = self._all_null_rows(JASlot)
        self.assertEqual(all_null_emp, [(4,)])
        self.assertEqual(all_null_dept, [(3,)])
        self.assertEqual(all_null_ticket, [(5,)])
        self.assertEqual(all_null_slot, [(2,)])

        # All-null rows are actually reachable via joins.
        self.assertIn(
            4,
            [row['manager_id'] for row in self.world[JAEmployee].values()])
        self.assertIn(
            3,
            [row['department_id']
             for row in self.world[JATicket].values()])

    def _relation_categories(self):
        emp_mgr = Relation(JAEmployee, 'manager_id', JAEmployee, 'id', 'mgr')
        emp_top = Relation(JAEmployee, 'manager_id', JAEmployee, 'id', 'top')
        ticket_dept = Relation(JATicket, 'department_id', JADepartment, 'id',
                               'department')
        dept_head = Relation(JADepartment, 'head_id', JAEmployee, 'code',
                             'head')
        ticket_assignee = Relation(JATicket, 'assignee_id', JAEmployee, 'id',
                                   'assignee')
        slot_pair = Relation(JASlot, 'pair_id', JAPair, 'label', 'pair')

        def walk(root_model, relations):
            counts = {k: 0 for k in
                      ('hit', 'null-fk', 'dangling', 'parent-miss')}
            for root_key in self.world[root_model]:
                nodes = resolve_path(self.world, root_model, root_key,
                                     relations)
                parent_present = True
                for idx, rel in enumerate(relations):
                    parent_row = nodes[idx][1]
                    node_row = nodes[idx + 1][1]
                    if not parent_present:
                        counts['parent-miss'] += 1
                    else:
                        fk_value = parent_row[rel.fk_col]
                        if fk_value is None:
                            counts['null-fk'] += 1
                        elif node_row is None:
                            counts['dangling'] += 1
                        else:
                            counts['hit'] += 1
                    parent_present = node_row is not None
            return counts

        return {
            'emp->mgr': walk(JAEmployee, [emp_mgr, emp_top]),
            'ticket->dept': walk(JATicket, [ticket_dept, dept_head]),
            'dept->head': walk(JADepartment, [dept_head]),
            'ticket->assignee': walk(JATicket, [ticket_assignee]),
            'slot->pair': walk(JASlot, [slot_pair]),
        }

    def _all_null_rows(self, model):
        pk_fields = model._meta.get_primary_keys()
        found = []
        for key, row in self.world[model].items():
            non_key = {col: val for col, val in row.items()
                       if col not in {f.column_name for f in pk_fields}}
            if all(val is None for val in non_key.values()):
                found.append(key)
        return sorted(found)

    # ------------------------------------------------------------------
    # assembly tests
    # ------------------------------------------------------------------
    def test_same_model_three_times_grouped_select(self):
        # Same model as root and twice as LEFT JOIN destination via aliases;
        # columns are selected model-grouped.
        Mgr = JAEmployee.alias('ja_mgr')
        Top = JAEmployee.alias('ja_top')
        query = (JAEmployee
                 .select(JAEmployee, Mgr, Top)
                 .join(Mgr, JOIN.LEFT_OUTER,
                       on=(JAEmployee.manager == Mgr.id).alias('mgr'))
                 .join(Top, JOIN.LEFT_OUTER,
                       on=(Mgr.manager == Top.id).alias('top'))
                 .order_by(JAEmployee.id))
        path = [
            Relation(JAEmployee, 'manager_id', JAEmployee, 'id', 'mgr'),
            Relation(JAEmployee, 'manager_id', JAEmployee, 'id', 'top'),
        ]
        raw_keys = self.root_keys(JAEmployee)
        with self.assertQueryCount(1):
            roots = list(query)
            self.assertGraph(roots, JAEmployee, [path], raw_keys)

        # All-NULL manager (employee 5 -> employee 4) materializes as an
        # object, it is not confused with a LEFT JOIN miss.
        emp5 = next(obj for obj in roots if obj.id == 5)
        self.assertIsNotNone(emp5.mgr)
        self.assertEqual(emp5.mgr.id, 4)
        self.assertIsNone(emp5.mgr.name)
        self.assertIsNone(emp5.mgr.code)
        self.assertIsNone(emp5.mgr.manager)
        # Dangling manager keeps its raw fk value while the relation is
        # None; the miss is attached as a plain attribute (the join used a
        # custom attr, not the fk descriptor).
        emp8 = next(obj for obj in roots if obj.id == 8)
        self.assertIsNone(emp8.mgr)
        self.assertEqual(emp8.__data__['manager'], 9901)
        self.assertIn('mgr', emp8.__dict__)
        self.assertIsNone(emp8.__dict__['mgr'])

    def test_same_model_three_times_interleaved_select(self):
        # The same query shape, but selected columns are interleaved so they
        # do not arrive grouped by model: strict round-robin plus a seeded
        # shuffle variant.
        Mgr = JAEmployee.alias('ja_mgr')
        Top = JAEmployee.alias('ja_top')
        roles = [(JAEmployee, JAEmployee._meta.sorted_fields),
                 (Mgr, Mgr.get_field_aliases()),
                 (Top, Top.get_field_aliases())]
        path = [
            Relation(JAEmployee, 'manager_id', JAEmployee, 'id', 'mgr'),
            Relation(JAEmployee, 'manager_id', JAEmployee, 'id', 'top'),
        ]
        # Round-robin: one field per role at a time.
        interleaved = []
        for idx in range(len(roles[0][1])):
            for _model, field_list in roles:
                interleaved.append(field_list[idx])
        query = (JAEmployee.select(*interleaved)
                 .join(Mgr, JOIN.LEFT_OUTER,
                       on=(JAEmployee.manager == Mgr.id).alias('mgr'))
                 .join(Top, JOIN.LEFT_OUTER,
                       on=(Mgr.manager == Top.id).alias('top'))
                 .order_by(JAEmployee.id))
        raw_keys = self.root_keys(JAEmployee)
        with self.assertQueryCount(1):
            self.assertGraph(list(query), JAEmployee, [path], raw_keys)

        # Seeded shuffle of the exact same field set.
        shuffled = list(interleaved)
        random.Random(SEED).shuffle(shuffled)
        query = (JAEmployee.select(*shuffled)
                 .join(Mgr, JOIN.LEFT_OUTER,
                       on=(JAEmployee.manager == Mgr.id).alias('mgr'))
                 .join(Top, JOIN.LEFT_OUTER,
                       on=(Mgr.manager == Top.id).alias('top'))
                 .order_by(JAEmployee.id))
        with self.assertQueryCount(1):
            self.assertGraph(list(query), JAEmployee, [path], raw_keys)

    def test_consecutive_left_joins_and_non_pk_fk(self):
        # Chain ticket -> department -> head (fk to JAEmployee.code), plus a
        # second branch ticket -> assignee (fk to the same model's pk).
        Head = JAEmployee.alias('ja_head')
        query = (JATicket
                 .select(JATicket, JADepartment, Head, JAEmployee)
                 .join(JADepartment, JOIN.LEFT_OUTER)
                 .join(Head, JOIN.LEFT_OUTER,
                       on=(JADepartment.head == Head.code).alias('head'))
                 .switch(JATicket)
                 .join(JAEmployee, JOIN.LEFT_OUTER,
                       on=(JATicket.assignee == JAEmployee.id)
                       .alias('assignee'))
                 .order_by(JATicket.id))
        chain = [
            Relation(JATicket, 'department_id', JADepartment, 'id',
                     'department'),
            Relation(JADepartment, 'head_id', JAEmployee, 'code', 'head'),
        ]
        branch = [
            Relation(JATicket, 'assignee_id', JAEmployee, 'id', 'assignee'),
        ]
        raw_keys = self.root_keys(JATicket)
        with self.assertQueryCount(1):
            roots = list(query)
            self.assertGraph(roots, JATicket, [chain, branch], raw_keys)

        # Ticket 3: all-null department still materializes and carries None
        # head; ticket 4: department found, head dangles -> None, raw code
        # stays on the department row.
        t3 = next(obj for obj in roots if obj.id == 3)
        self.assertIsNotNone(t3.department)
        self.assertEqual(t3.department.id, 3)
        self.assertIsNone(t3.department.title)
        self.assertIsNone(t3.department.head)
        t4 = next(obj for obj in roots if obj.id == 4)
        self.assertIsNotNone(t4.department)
        self.assertIsNone(t4.department.head)
        self.assertEqual(t4.department.__data__['head'], 'E999')
        self.assertIn('head', t4.department.__rel__)
        self.assertIsNone(t4.department.__rel__['head'])
        # Ticket 7: both fks dangle; ticket 5: all-null ticket itself.
        t7 = next(obj for obj in roots if obj.id == 7)
        self.assertIsNone(t7.department)
        self.assertEqual(t7.__data__['department'], 9903)
        self.assertIsNone(t7.assignee)
        self.assertEqual(t7.__data__['assignee'], 9902)
        t5 = next(obj for obj in roots if obj.id == 5)
        self.assertEqual(t5.__data__['note'], None)
        self.assertIsNone(t5.department)
        self.assertIsNone(t5.assignee)

    def test_consecutive_left_joins_interleaved_select(self):
        # The chain again, but columns from the four roles are interleaved,
        # including duplicated column names (id/title/name) side by side.
        Head = JAEmployee.alias('ja_head')
        roles = [(JATicket, JATicket._meta.sorted_fields),
                 (JADepartment, JADepartment._meta.sorted_fields),
                 (Head, Head.get_field_aliases()),
                 (JAEmployee, JAEmployee._meta.sorted_fields)]
        interleaved = []
        width = max(len(fields) for _m, fields in roles)
        for idx in range(width):
            for _model, field_list in roles:
                if idx < len(field_list):
                    interleaved.append(field_list[idx])
        query = (JATicket.select(*interleaved)
                 .join(JADepartment, JOIN.LEFT_OUTER)
                 .join(Head, JOIN.LEFT_OUTER,
                       on=(JADepartment.head == Head.code).alias('head'))
                 .switch(JATicket)
                 .join(JAEmployee, JOIN.LEFT_OUTER,
                       on=(JATicket.assignee == JAEmployee.id)
                       .alias('assignee'))
                 .order_by(JATicket.id))
        chain = [
            Relation(JATicket, 'department_id', JADepartment, 'id',
                     'department'),
            Relation(JADepartment, 'head_id', JAEmployee, 'code', 'head'),
        ]
        branch = [
            Relation(JATicket, 'assignee_id', JAEmployee, 'id', 'assignee'),
        ]
        raw_keys = self.root_keys(JATicket)
        with self.assertQueryCount(1):
            roots = list(query)
            self.assertGraph(roots, JATicket, [chain, branch], raw_keys)

    def test_composite_pk_fk_to_non_primary_key(self):
        # JASlot.pair references JAPair.label; JAPair has a composite pk.
        Pair = JAPair.alias('ja_pair')
        query = (JASlot.select(JASlot, Pair)
                 .join(Pair, JOIN.LEFT_OUTER,
                       on=(JASlot.pair == Pair.label).alias('pair'))
                 .order_by(JASlot.id))
        path = [Relation(JASlot, 'pair_id', JAPair, 'label', 'pair')]
        raw_keys = self.root_keys(JASlot)
        with self.assertQueryCount(1):
            roots = list(query)
            self.assertGraph(roots, JASlot, [path], raw_keys)

        s2 = next(obj for obj in roots if obj.id == 2)
        # Exists with all non-key NULL; its own pair fk is NULL -> miss None.
        self.assertIsNone(s2.tag)
        self.assertIsNone(s2.pair)
        s4 = next(obj for obj in roots if obj.id == 4)
        self.assertIsNone(s4.pair)
        self.assertEqual(s4.__data__['pair'], 'P999')
        self.assertIn('pair', s4.__rel__)
        self.assertIsNone(s4.__rel__['pair'])

    def test_many_scrambled_permutations_match_reference(self):
        # Re-emit the four-role chain with several seeded column permutations.
        # Each permutation must reconstruct the identical reference graph, so
        # a misrouted column can never hide behind one lucky ordering.
        Head = JAEmployee.alias('ja_head')
        roles = [(JATicket, JATicket._meta.sorted_fields),
                 (JADepartment, JADepartment._meta.sorted_fields),
                 (Head, Head.get_field_aliases()),
                 (JAEmployee, JAEmployee._meta.sorted_fields)]
        all_fields = [field for _model, field_list in roles
                      for field in field_list]
        chain = [
            Relation(JATicket, 'department_id', JADepartment, 'id',
                     'department'),
            Relation(JADepartment, 'head_id', JAEmployee, 'code', 'head'),
        ]
        branch = [
            Relation(JATicket, 'assignee_id', JAEmployee, 'id', 'assignee'),
        ]
        raw_keys = self.root_keys(JATicket)
        perm_rng = random.Random(SEED)
        for _ in range(5):
            permuted = list(all_fields)
            perm_rng.shuffle(permuted)
            query = (JATicket.select(*permuted)
                     .join(JADepartment, JOIN.LEFT_OUTER)
                     .join(Head, JOIN.LEFT_OUTER,
                           on=(JADepartment.head == Head.code).alias('head'))
                     .switch(JATicket)
                     .join(JAEmployee, JOIN.LEFT_OUTER,
                           on=(JATicket.assignee == JAEmployee.id)
                           .alias('assignee'))
                     .order_by(JATicket.id))
            with self.assertQueryCount(1):
                roots = list(query)
                self.assertGraph(roots, JATicket, [chain, branch], raw_keys)

    def test_partial_columns_all_null_row_vs_miss(self):
        # Selecting the pk disambiguates a true miss from an existing row
        # whose non-key columns are all NULL.
        Mgr = JAEmployee.alias('ja_mgr')
        query = (JAEmployee
                 .select(JAEmployee.id, JAEmployee.manager, Mgr.id, Mgr.name)
                 .join(Mgr, JOIN.LEFT_OUTER,
                       on=(JAEmployee.manager == Mgr.id).alias('mgr'))
                 .order_by(JAEmployee.id))
        rows = list(query)
        emp4_mgr = next(obj for obj in rows if obj.id == 5).mgr
        self.assertIsNotNone(emp4_mgr)       # employee 4 exists
        self.assertEqual(emp4_mgr.id, 4)
        self.assertEqual(emp4_mgr.__data__['name'], None)
        emp8_mgr = next(obj for obj in rows if obj.id == 8).mgr
        self.assertIsNone(emp8_mgr)          # dangling 9901: true miss
