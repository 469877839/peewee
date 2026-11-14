"""
Independent-reference tests for join-result object assembly.

The expected object graphs in this module are never produced by peewee's
cursor wrappers. They are recomputed by ``ReferenceOracle`` from:

1. the raw rows fetched from the sqlite cursor (``cursor.description`` +
   ``fetchall()``), and
2. the join relationships declared up-front by the test (which source key
   carries which fk column, which destination key matches on which column,
   outer vs inner, and the attribute used for attachment).

Coverage (fixed, explicit seed data, no RNG; runs against an in-memory
sqlite database and finishes in well under a second):

* the same model joined twice in one query through two foreign keys
  (un-aliased model + ``ModelAlias``);
* consecutive LEFT JOINs, including a missing parent / grandparent branch;
* a foreign key that points at a non-primary unique column, including a
  dangling fk value;
* a projection whose columns are not grouped by model (interleaved);
* a row that really exists but whose every non-key column is NULL, kept
  distinct from a genuinely unmatched LEFT JOIN side (None);
* a matched row whose only selected destination cells are all NULL (the
  join key is not projected), which must still hydrate as a real related
  instance rather than collapse to None.

The oracle never imports or calls peewee's cursor wrappers. Its only inputs
are the raw DB-API cursor rows and the relationships declared by each test;
when a join key is omitted from the projection it is resolved from a
separate, plain ``SELECT *`` on the physical table.
"""
import unittest

from peewee import *

from .base import get_in_memory_db
from .base import TestModel


database = get_in_memory_db()


class _OracleModel(TestModel):
    pass


# ---------------------------------------------------------------------------
# Fixture models
# ---------------------------------------------------------------------------

class Party(_OracleModel):
    # ``code`` is a unique non-primary key: other models reference this column
    # rather than the integer primary key.
    code = CharField(unique=True)
    label = CharField(null=True)


class Membership(_OracleModel):
    # Two foreign keys to the same model. ``inviter`` is nullable so LEFT
    # JOIN unmatched sides can be exercised.
    member = ForeignKeyField(Party, field='code')
    inviter = ForeignKeyField(Party, field='code', null=True,
                              backref='invitations')
    role = CharField(null=True)


class Org(_OracleModel):
    # Self-referential chain for consecutive LEFT JOINs. A leaf has no
    # parent; deeper missing intermediaries are seeded explicitly.
    name = CharField()
    parent = ForeignKeyField('self', null=True, backref='children')
    note = CharField(null=True)


class WeakTarget(_OracleModel):
    # Primary key is a non-NULL unique CharField, but the only *payload*
    # column is nullable. When a projection selects just the payload column
    # (not the key), a matched row can still hydrate to all-NULL cells.
    code = CharField(primary_key=True)
    detail = CharField(null=True)


class WeakRef(_OracleModel):
    name = CharField()
    target = ForeignKeyField(WeakTarget, field='code', null=True,
                             backref='weak_refs')


MODELS = [Party, Membership, Org, WeakTarget, WeakRef]


# ---------------------------------------------------------------------------
# Independent reference oracle
# ---------------------------------------------------------------------------

class _ExpectedInstance:
    """Plain stand-in for a hydrated model: identity is per (key, row)."""
    def __init__(self, key):
        self._key = key
        self.data = {}
        self.rels = {}

    def __repr__(self):
        return '_ExpectedInstance(%r, data=%r, rels=%r)' % (
            self._key, self.data, self.rels)


def load_table_index(model, key_column):
    """Load a table through a raw sqlite SELECT, indexed by a column.

    This is independent of peewee's join assembler: it reads the physical
    table directly and returns ``{column_name: {value: row_dict}}``.
    """
    cursor = database.execute_sql(
        'SELECT * FROM "%s"' % model._meta.table_name)
    names = [d[0] for d in cursor.description]
    index = {name: {} for name in names}
    for raw in cursor.fetchall():
        row = dict(zip(names, raw))
        for name in names:
            index[name][row[name]] = row
    return index


class ReferenceOracle:
    """Reconstruct expected object graphs without peewee's assembler.

    Parameters
    ----------
    cursor:
        An executed DB-API cursor. Its ``description`` names are matched
        against ``projection`` positionally.
    projection:
        Mapping ``column index -> (source key, field name)``. The source key
        is whatever identity the test uses for a joined source (a model class
        or a ``ModelAlias`` instance).
    root_key:
        Source key whose instances form the returned, row-aligned list.
    edges:
        Iterable of declared joins, each a tuple of
        ``(src_key, attr, dest_key, src_fk_column, dest_match_column,
        is_outer, attach_kind)`` where ``attach_kind`` is 'fk' when the
        attribute is a foreign-key field on the source, 'attr' for a plain
        declared attribute, and 'backref' for the reverse collection.
    """

    def __init__(self, cursor, projection, root_key, edges,
                 source_tables=None):
        self.description = [d[0] for d in cursor.description]
        self.raw_rows = cursor.fetchall()
        self.projection = projection
        self.root_key = root_key
        self.edges = edges
        # Optional ``{source key: {match column: {value: row_dict}}}`` built
        # independently from plain per-table SELECTs. It lets the oracle
        # resolve a join key that the main projection does not include, so
        # presence never depends on whether a key column happens to be
        # selected.
        self.source_tables = source_tables or {}

    @staticmethod
    def _keys_match(left, right):
        # SQL join semantics used by the fixtures: NULL = NULL does not match
        # for the explicit ON predicates here (sqlite's default "=" behavior),
        # so both must be non-None and equal.
        return left is not None and right is not None and left == right

    def build(self):
        results = []
        for raw in self.raw_rows:
            instances = {}
            present = set()

            # 1. Hydrate one instance per selected source from raw cell
            #    values, using the positional projection only.
            for idx, (key, field_name) in self.projection.items():
                value = raw[idx]
                inst = instances.get(key)
                if inst is None:
                    inst = instances[key] = _ExpectedInstance(key)
                inst.data[field_name] = value
                if value is not None:
                    present.add(key)

            # 2. Apply the declared joins. Match/miss is decided from the
            #    declared key columns on the raw row, never from the code
            #    under test.
            for (src_key, attr, dest_key, fk_col, match_col,
                 is_outer, kind) in self.edges:
                src = instances.get(src_key)
                dest = instances.get(dest_key)
                dest_has_non_null = dest_key in present

                if kind == 'backref':
                    if src is not None and dest_has_non_null:
                        src.rels.setdefault(attr, []).append(dest)
                    continue

                if src is None:
                    continue

                fk_value = src.data.get(fk_col) if fk_col else None
                hydrated_match = (dest.data.get(match_col)
                                  if dest is not None and match_col
                                  else None)
                # Independently confirm the destination row exists for this
                # fk value, even when the key was not projected.
                table = self.source_tables.get(dest_key)
                table_match = None
                if table is not None and match_col in table and \
                        fk_value is not None:
                    table_match = table[match_col].get(fk_value)

                predicate_matches = self._keys_match(fk_value,
                                                      hydrated_match)
                if not predicate_matches and table_match is not None:
                    predicate_matches = True

                dest_present = dest_has_non_null or predicate_matches

                if not dest_present:
                    if is_outer:
                        # Verified-absent side: None, while the raw fk value
                        # stays on the source's data.
                        src.rels[attr] = None
                    continue

                # Fill an un-projected key from the independent table so
                # downstream edges and snapshots see the real join value.
                if (dest is not None and match_col and
                        dest.data.get(match_col) is None and
                        table_match is not None):
                    dest.data[match_col] = table_match[match_col]

                src.rels[attr] = dest

            results.append(instances[self.root_key])
        return results


def _model_snapshot(instance, field_names, rel_specs):
    """Serializable (field values, relation values) view of a peewee object.

    ``rel_specs`` maps attribute name to either:
    * a tuple of field names, compared recursively for model relations,
    * a list of field names, compared as a backref collection, or
    * ``None``, meaning the relation must be ``None``.
    """
    data = {name: getattr(instance, name) for name in field_names}
    rels = {}
    for attr, nested in rel_specs.items():
        value = getattr(instance, attr)
        if nested is None:
            rels[attr] = None if value is None else value
        elif isinstance(nested, list):
            if value is None:
                rels[attr] = None
            else:
                rels[attr] = [{n: getattr(item, n) for n in nested}
                              for item in value]
        else:
            if value is None:
                rels[attr] = None
            else:
                rels[attr] = {n: getattr(value, n) for n in nested}
    return data, rels


def _expected_snapshot(expected, field_names_by_key, rel_specs):
    data = {k: v for k, v in expected.data.items()
            if k in field_names_by_key[expected._key]}
    rels = {}
    for attr, nested in rel_specs.items():
        value = expected.rels.get(attr)
        if nested is None:
            rels[attr] = None if value is None else value
        elif isinstance(nested, list):
            if value is None:
                rels[attr] = None
            else:
                rels[attr] = [{n: item.data.get(n) for n in nested}
                              for item in value]
        else:
            if value is None:
                rels[attr] = None
            else:
                rels[attr] = {n: value.data.get(n) for n in nested}
    return data, rels


class JoinAssemblyOracleTestCase(unittest.TestCase):
    def setUp(self):
        # Bind the fixture models to an isolated in-memory sqlite database.
        self._bind_ctx = database.bind_ctx(MODELS, bind_refs=False,
                                           bind_backrefs=False)
        self._bind_ctx.__enter__()
        database.connect(reuse_if_open=True)
        database.drop_tables(MODELS, safe=True)
        database.create_tables(MODELS)

    def tearDown(self):
        try:
            database.drop_tables(MODELS, safe=True)
        finally:
            database.close()
            self._bind_ctx.__exit__(None, None, None)

    def seed_parties(self):
        # 'C' has every non-key column NULL but the row really exists.
        for code, label in (('A', 'alpha'), ('B', 'bravo'),
                            ('C', None), ('D', 'delta')):
            Party.create(code=code, label=label)

    def test_same_model_joined_twice(self):
        self.seed_parties()
        Membership.create(member='A', inviter='B', role='member')
        Membership.create(member='B', inviter=None, role='solo')
        # Dangling inviter code: no Party row with code 'Z'.
        Membership.create(member='C', inviter='Z', role='ghost')

        Inviter = Party.alias()
        query = (Membership
                 .select(Membership, Party, Inviter)
                 .join(Party, on=(Membership.member == Party.code))
                 .switch(Membership)
                 .join(Inviter, JOIN.LEFT_OUTER,
                       on=(Membership.inviter == Inviter.code))
                 .order_by(Membership.id))

        # Raw DB-API cursor: the oracle reads original sqlite rows directly,
        # never touching peewee's model cursor wrapper / assembler.
        cursor = database.execute_sql(*query.sql())
        projection = {}
        idx = 0
        for key in (Membership, Party, Inviter):
            model = key.model if isinstance(key, ModelAlias) else key
            for field in model._meta.sorted_fields:
                projection[idx] = (key, field.safe_name)
                idx += 1

        edges = [
            (Membership, 'member', Party, 'member_id', 'code', False, 'fk'),
            (Membership, 'inviter', Inviter, 'inviter_id', 'code',
             True, 'fk'),
        ]
        expected = ReferenceOracle(cursor, projection, Membership,
                                   edges).build()

        actual = list(query)
        self.assertEqual(len(actual), len(expected))

        party_fields = ('id', 'code', 'label')
        member_fields = ('id', 'member_id', 'inviter_id', 'role')
        rel_specs = {'member': party_fields, 'inviter': party_fields}
        field_map = {Membership: member_fields, Party: party_fields,
                     Inviter: party_fields}
        for got, want in zip(actual, expected):
            got_data, got_rels = _model_snapshot(
                got, member_fields, rel_specs)
            want_data, want_rels = _expected_snapshot(
                want, field_map, rel_specs)
            self.assertEqual(got_data, want_data)
            self.assertEqual(got_rels, want_rels)

        by_role = {m.role: m for m in actual}
        self.assertEqual(by_role['member'].member.code, 'A')
        self.assertEqual(by_role['member'].inviter.code, 'B')
        self.assertIsNone(by_role['solo'].inviter)
        # Dangling fk: the id column is retained, the relation is None.
        self.assertEqual(by_role['ghost'].inviter_id, 'Z')
        self.assertIsNone(by_role['ghost'].inviter)

    def test_consecutive_left_joins(self):
        root = Org.create(name='root', parent=None)
        mid = Org.create(name='mid', parent=root)
        Org.create(name='leaf', parent=mid)
        Org.create(name='orphan', parent=None)

        Parent = Org.alias('parent_org')
        Grandparent = Org.alias('grandparent_org')
        query = (Org
                 .select(Org, Parent, Grandparent)
                 .join(Parent, JOIN.LEFT_OUTER,
                       on=(Org.parent == Parent.id))
                 .join(Grandparent, JOIN.LEFT_OUTER, src=Parent,
                       on=(Parent.parent == Grandparent.id))
                 .order_by(Org.id))

        # Raw DB-API cursor: the oracle reads original sqlite rows directly,
        # never touching peewee's model cursor wrapper / assembler.
        cursor = database.execute_sql(*query.sql())
        projection = {}
        idx = 0
        for key in (Org, Parent, Grandparent):
            model = key.model if isinstance(key, ModelAlias) else key
            for field in model._meta.sorted_fields:
                projection[idx] = (key, field.safe_name)
                idx += 1

        edges = [
            (Org, 'parent', Parent, 'parent_id', 'id', True, 'fk'),
            (Parent, 'parent', Grandparent, 'parent_id', 'id',
             True, 'fk'),
        ]
        expected = ReferenceOracle(cursor, projection, Org, edges).build()
        actual = list(query)

        org_fields = ('id', 'name', 'parent_id', 'note')
        rel_specs = {'parent': org_fields}
        self.assertEqual([o.name for o in actual],
                         ['root', 'mid', 'leaf', 'orphan'])

        field_map = {Org: org_fields, Parent: org_fields,
                     Grandparent: org_fields}
        for got, want in zip(actual, expected):
            got_data, got_rels = _model_snapshot(
                got, org_fields, rel_specs)
            want_data, want_rels = _expected_snapshot(
                want, field_map, rel_specs)
            self.assertEqual(got_data, want_data)
            self.assertEqual(got_rels, want_rels)

        by_name = {o.name: o for o in actual}
        want_by_name = {e.data['name']: e for e in expected}
        self.assertIsNone(by_name['root'].parent)
        self.assertEqual(by_name['mid'].parent.name, 'root')
        # leaf -> mid -> root: the grandparent is reached through the
        # intermediary, proving two chained outer joins wire correctly and
        # nothing is attached at the wrong level.
        self.assertEqual(by_name['leaf'].parent.name, 'mid')
        self.assertEqual(by_name['leaf'].parent.parent.name, 'root')
        # Same two-level path, independently derived by the oracle.
        leaf_want = want_by_name['leaf']
        self.assertEqual(leaf_want.rels['parent'].data['name'], 'mid')
        self.assertEqual(
            leaf_want.rels['parent'].rels['parent'].data['name'], 'root')
        self.assertIsNone(by_name['orphan'].parent)

    def test_foreign_key_to_non_primary_column(self):
        self.seed_parties()
        m1 = Membership.create(member='A', inviter='D', role='x')
        Membership.create(member='D', inviter='A', role='y')

        query = (Membership
                 .select(Membership, Party)
                 .join(Party, on=(Membership.member == Party.code))
                 .order_by(Membership.id))
        # Raw DB-API cursor: the oracle reads original sqlite rows directly,
        # never touching peewee's model cursor wrapper / assembler.
        cursor = database.execute_sql(*query.sql())

        projection = {}
        idx = 0
        for key in (Membership, Party):
            for field in key._meta.sorted_fields:
                projection[idx] = (key, field.safe_name)
                idx += 1
        edges = [(Membership, 'member', Party, 'member_id', 'code',
                  False, 'fk')]
        expected = ReferenceOracle(cursor, projection, Membership,
                                   edges).build()
        actual = list(query)

        member_fields = ('id', 'member_id', 'inviter_id', 'role')
        party_fields = ('id', 'code', 'label')
        field_map = {Membership: member_fields, Party: party_fields}
        rel_specs = {'member': party_fields}
        for got, want in zip(actual, expected):
            got_data, got_rels = _model_snapshot(
                got, member_fields, rel_specs)
            want_data, want_rels = _expected_snapshot(
                want, field_map, rel_specs)
            self.assertEqual(got_data, want_data)
            self.assertEqual(got_rels, want_rels)

        first = actual[0]
        self.assertEqual(first.id, m1.id)
        self.assertEqual(first.member.code, 'A')
        # The fk stores the code, not Party.id.
        self.assertEqual(first.member_id, 'A')
        self.assertEqual(first.member.label, 'alpha')

    def test_interleaved_projection_columns(self):
        self.seed_parties()
        Membership.create(member='A', inviter='B', role='r1')
        Membership.create(member='C', inviter=None, role='r2')

        Inviter = Party.alias('inviter_party')
        # Deliberately interleave fields from the three sources instead of
        # passing whole models.
        query = (Membership
                 .select(Membership.role,
                         Party.code,
                         Membership.id,
                         Inviter.label,
                         Party.label,
                         Inviter.code,
                         Membership.member,
                         Membership.inviter)
                 .join(Party, on=(Membership.member == Party.code))
                 .switch(Membership)
                 .join(Inviter, JOIN.LEFT_OUTER,
                       on=(Membership.inviter == Inviter.code))
                 .order_by(Membership.id))

        # Raw DB-API cursor: the oracle reads original sqlite rows directly,
        # never touching peewee's model cursor wrapper / assembler.
        cursor = database.execute_sql(*query.sql())
        projection = {}
        for idx, node in enumerate(query._returning):
            node = node.unwrap()
            if isinstance(node, FieldAlias):
                # safe_name matches the attribute the assembler writes (for a
                # foreign key this is the object-id name, e.g. member_id).
                key, field_name = node.source, node.safe_name
            else:
                key, field_name = node.model, node.safe_name
            projection[idx] = (key, field_name)

        edges = [
            (Membership, 'member', Party, 'member_id', 'code', False, 'fk'),
            (Membership, 'inviter', Inviter, 'inviter_id', 'code',
             True, 'fk'),
        ]
        expected = ReferenceOracle(cursor, projection, Membership,
                                   edges).build()
        actual = list(query)

        member_fields = ('id', 'member_id', 'inviter_id', 'role')
        party_fields = ('code', 'label')
        rel_specs = {'member': party_fields, 'inviter': party_fields}
        field_map = {Membership: member_fields, Party: party_fields,
                     Inviter: party_fields}
        for got, want in zip(actual, expected):
            got_data, got_rels = _model_snapshot(
                got, member_fields, rel_specs)
            want_data, want_rels = _expected_snapshot(
                want, field_map, rel_specs)
            self.assertEqual(got_data, want_data)
            self.assertEqual(got_rels, want_rels)

        first, second = actual
        self.assertEqual(first.role, 'r1')
        self.assertEqual(first.member.code, 'A')
        self.assertEqual(first.inviter.code, 'B')
        self.assertEqual(second.role, 'r2')
        self.assertEqual(second.member.code, 'C')
        self.assertIsNone(second.inviter)

    def test_all_null_related_row_is_not_none(self):
        self.seed_parties()  # Party 'C' exists with label = NULL.
        Membership.create(member='C', inviter='C', role='nullish')

        Inviter = Party.alias('inviter_party')
        query = (Membership
                 .select(Membership, Party, Inviter)
                 .join(Party, on=(Membership.member == Party.code))
                 .switch(Membership)
                 .join(Inviter, JOIN.LEFT_OUTER,
                       on=(Membership.inviter == Inviter.code)))
        # Raw DB-API cursor: the oracle reads original sqlite rows directly,
        # never touching peewee's model cursor wrapper / assembler.
        cursor = database.execute_sql(*query.sql())

        projection = {}
        idx = 0
        for key in (Membership, Party, Inviter):
            model = key.model if isinstance(key, ModelAlias) else key
            for field in model._meta.sorted_fields:
                projection[idx] = (key, field.safe_name)
                idx += 1
        edges = [
            (Membership, 'member', Party, 'member_id', 'code', False, 'fk'),
            (Membership, 'inviter', Inviter, 'inviter_id', 'code',
             True, 'fk'),
        ]
        expected = ReferenceOracle(cursor, projection, Membership,
                                   edges).build()
        actual = list(query)

        self.assertEqual(len(actual), 1)
        got = actual[0]
        want = expected[0]

        # Both relations point at the real 'C' row: present, not None, even
        # though every non-key column is NULL.
        self.assertIsNotNone(got.member)
        self.assertIsNotNone(got.inviter)
        self.assertEqual(got.member.code, 'C')
        self.assertEqual(got.inviter.code, 'C')
        self.assertIsNone(got.member.label)
        self.assertIsNone(got.inviter.label)

        party_fields = ('id', 'code', 'label')
        member_fields = ('id', 'member_id', 'inviter_id', 'role')
        rel_specs = {'member': party_fields, 'inviter': party_fields}
        got_data, got_rels = _model_snapshot(
            got, member_fields, rel_specs)
        want_data, want_rels = _expected_snapshot(
            want,
            {Membership: member_fields, Party: party_fields,
             Inviter: party_fields},
            rel_specs)
        self.assertEqual(got_data, want_data)
        self.assertEqual(got_rels, want_rels)

    def test_outer_join_matched_row_hydrates_to_all_null_cells(self):
        # The target row really exists and is matched through the foreign
        # key, but the projection selects only its nullable payload column,
        # which is NULL. So every selected target cell is NULL. The
        # assembler must still attach the real instance (and not treat the
        # outer join as unmatched), since the source fk is non-NULL and the
        # database evaluated the join.
        target = WeakTarget.create(code='T1', detail=None)
        other = WeakTarget.create(code='T2', detail='present')
        WeakRef.create(name='ref-null-payload', target=target)
        WeakRef.create(name='ref-with-payload', target=other)
        # A NULL fk is a genuinely unmatched outer-join side -> None.
        WeakRef.create(name='ref-null-fk', target=None)

        # Interleaved, non-model-grouped projection. Note WeakTarget.code
        # (the join key) is deliberately NOT selected.
        query = (WeakRef
                 .select(WeakRef.name,
                         WeakTarget.detail,
                         WeakRef.id,
                         WeakRef.target)
                 .join(WeakTarget, JOIN.LEFT_OUTER,
                       on=(WeakRef.target == WeakTarget.code))
                 .order_by(WeakRef.id))
        cursor = database.execute_sql(*query.sql())

        projection = {}
        for idx, node in enumerate(query._returning):
            node = node.unwrap()
            if isinstance(node, FieldAlias):
                key, field_name = node.source, node.safe_name
            else:
                key, field_name = node.model, node.safe_name
            projection[idx] = (key, field_name)

        # The oracle resolves existence independently from the physical
        # target table, so it does not mistake an all-NULL matched slice for
        # an unmatched join.
        target_index = load_table_index(WeakTarget, 'code')
        source_tables = {WeakTarget: target_index}
        edges = [(WeakRef, 'target', WeakTarget, 'target_id', 'code',
                  True, 'fk')]
        expected = ReferenceOracle(cursor, projection, WeakRef,
                                   edges,
                                   source_tables=source_tables).build()
        actual = list(query)

        self.assertEqual(len(actual), 3)
        null_payload, with_payload, null_fk = actual

        # Matched all-NULL-cells target: present, not None.
        self.assertIsNotNone(null_payload.target)
        self.assertIsNone(null_payload.target.detail)
        self.assertEqual(null_payload.target_id, 'T1')
        # A normal payload row still wires to the same attribute.
        self.assertEqual(with_payload.target.detail, 'present')
        # NULL fk: genuinely unmatched outer side -> None.
        self.assertIsNone(null_fk.target_id)
        self.assertIsNone(null_fk.target)

        ref_fields = ('id', 'name', 'target_id')
        rel_specs = {'target': ('detail',)}
        field_map = {WeakRef: ref_fields, WeakTarget: ('detail',)}
        for got, want in zip(actual, expected):
            got_data, got_rels = _model_snapshot(
                got, ref_fields, rel_specs)
            want_data, want_rels = _expected_snapshot(
                want, field_map, rel_specs)
            self.assertEqual(got_data, want_data)
            self.assertEqual(got_rels, want_rels)

    def test_unmatched_outer_side_is_none(self):
        self.seed_parties()
        # inviter NULL => the outer side is genuinely unmatched.
        Membership.create(member='A', inviter=None, role='no-inviter')

        Inviter = Party.alias('inviter_party')
        query = (Membership
                 .select(Membership, Party, Inviter)
                 .join(Party, on=(Membership.member == Party.code))
                 .switch(Membership)
                 .join(Inviter, JOIN.LEFT_OUTER,
                       on=(Membership.inviter == Inviter.code)))
        # Raw DB-API cursor: the oracle reads original sqlite rows directly,
        # never touching peewee's model cursor wrapper / assembler.
        cursor = database.execute_sql(*query.sql())

        projection = {}
        idx = 0
        for key in (Membership, Party, Inviter):
            model = key.model if isinstance(key, ModelAlias) else key
            for field in model._meta.sorted_fields:
                projection[idx] = (key, field.safe_name)
                idx += 1
        edges = [
            (Membership, 'member', Party, 'member_id', 'code', False, 'fk'),
            (Membership, 'inviter', Inviter, 'inviter_id', 'code',
             True, 'fk'),
        ]
        expected = ReferenceOracle(cursor, projection, Membership,
                                   edges).build()
        actual = list(query)

        got = actual[0]
        want = expected[0]
        self.assertIsNone(got.inviter)
        self.assertIsNone(got.inviter_id)
        self.assertIsNotNone(got.member)
        self.assertEqual(got.member.code, 'A')

        party_fields = ('id', 'code', 'label')
        member_fields = ('id', 'member_id', 'inviter_id', 'role')
        rel_specs = {'member': party_fields, 'inviter': party_fields}
        _, got_rels = _model_snapshot(got, member_fields, rel_specs)
        _, want_rels = _expected_snapshot(
            want,
            {Membership: member_fields, Party: party_fields,
             Inviter: party_fields},
            rel_specs)
        self.assertEqual(got_rels, want_rels)


if __name__ == '__main__':
    unittest.main()
