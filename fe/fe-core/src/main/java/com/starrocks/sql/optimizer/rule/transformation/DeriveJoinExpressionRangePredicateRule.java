// Copyright 2021-present StarRocks, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.starrocks.sql.optimizer.rule.transformation;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.starrocks.sql.ast.expression.BinaryType;
import com.starrocks.type.Type;
import com.starrocks.sql.optimizer.ExpressionContext;
import com.starrocks.sql.optimizer.OptExpression;
import com.starrocks.sql.optimizer.OptimizerContext;
import com.starrocks.sql.optimizer.Utils;
import com.starrocks.sql.optimizer.base.ColumnRefFactory;
import com.starrocks.sql.optimizer.base.ColumnRefSet;
import com.starrocks.sql.optimizer.operator.Operator;
import com.starrocks.sql.optimizer.operator.OperatorBuilderFactory;
import com.starrocks.sql.optimizer.operator.OperatorType;
import com.starrocks.sql.optimizer.operator.logical.LogicalJoinOperator;
import com.starrocks.sql.optimizer.operator.logical.LogicalOperator;
import com.starrocks.sql.optimizer.operator.logical.LogicalProjectOperator;
import com.starrocks.sql.optimizer.operator.logical.LogicalScanOperator;
import com.starrocks.sql.optimizer.operator.pattern.Pattern;
import com.starrocks.sql.optimizer.operator.scalar.BinaryPredicateOperator;
import com.starrocks.sql.optimizer.operator.scalar.CastOperator;
import com.starrocks.sql.optimizer.operator.scalar.ColumnRefOperator;
import com.starrocks.sql.optimizer.operator.scalar.ConstantOperator;
import com.starrocks.sql.optimizer.operator.scalar.OperatorFunctionChecker;
import com.starrocks.sql.optimizer.operator.scalar.ScalarOperator;
import com.starrocks.sql.optimizer.rewrite.ReplaceColumnRefRewriter;
import com.starrocks.sql.optimizer.rewrite.ScalarOperatorRewriter;
import com.starrocks.sql.optimizer.rewrite.scalar.FoldConstantsRule;
import com.starrocks.sql.optimizer.rewrite.scalar.ImplicitCastRule;
import com.starrocks.sql.optimizer.rewrite.scalar.ReduceCastRule;
import com.starrocks.sql.optimizer.rewrite.scalar.ScalarOperatorRewriteRule;
import com.starrocks.sql.optimizer.rule.RuleType;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static com.starrocks.sql.optimizer.operator.OpRuleBit.OP_PARTITION_PRUNED;

/**
 * Derives redundant scan-side predicates from an inner-join equality where one side is a base column
 * and the other side is a monotonic expression over a bounded column from the opposite side.
 *
 * Example:
 *   fact.datamonth = date_format(cast(event.datadate as date), '%Y%m')
 *   event.datadate between '20260705' and '20260708'
 * derives:
 *   fact.datamonth >= 202607 and fact.datamonth <= 202607
 */
public class DeriveJoinExpressionRangePredicateRule extends TransformationRule {
    private static final List<ScalarOperatorRewriteRule> CONSTANT_REWRITE_RULES = Lists.newArrayList(
            new ImplicitCastRule(),
            new ReduceCastRule(),
            new FoldConstantsRule(true)
    );

    public DeriveJoinExpressionRangePredicateRule() {
        super(RuleType.TF_DERIVE_JOIN_EXPRESSION_RANGE_PREDICATE,
                Pattern.create(OperatorType.LOGICAL_JOIN, OperatorType.PATTERN_LEAF, OperatorType.PATTERN_LEAF));
    }

    @Override
    public List<OptExpression> transform(OptExpression input, OptimizerContext context) {
        LogicalJoinOperator join = input.getOp().cast();
        if (!join.getJoinType().isInnerJoin() || join.getOnPredicate() == null) {
            return Collections.emptyList();
        }

        OptExpression leftChild = input.inputAt(0);
        OptExpression rightChild = input.inputAt(1);
        ColumnRefSet leftOutputColumns = leftChild.getOutputColumns();
        ColumnRefSet rightOutputColumns = rightChild.getOutputColumns();
        ColumnRefFactory columnRefFactory = context.getColumnRefFactory();

        Map<ColumnRefOperator, ScalarOperator> leftLineage = getLineage(leftChild, columnRefFactory);
        Map<ColumnRefOperator, ScalarOperator> rightLineage = getLineage(rightChild, columnRefFactory);
        Map<ColumnRefOperator, ColumnBounds> leftBounds = collectColumnBounds(leftChild, columnRefFactory);
        Map<ColumnRefOperator, ColumnBounds> rightBounds = collectColumnBounds(rightChild, columnRefFactory);
        Set<String> leftExistingPredicates = collectPredicateKeys(leftChild);
        Set<String> rightExistingPredicates = collectPredicateKeys(rightChild);

        List<ScalarOperator> leftPredicates = Lists.newArrayList();
        List<ScalarOperator> rightPredicates = Lists.newArrayList();
        for (ScalarOperator conjunct : Utils.extractConjuncts(join.getOnPredicate())) {
            if (!(conjunct instanceof BinaryPredicateOperator)) {
                continue;
            }
            BinaryPredicateOperator binary = (BinaryPredicateOperator) conjunct;
            if (!BinaryType.EQ.equals(binary.getBinaryType())) {
                continue;
            }

            ScalarOperator lhs = binary.getChild(0);
            ScalarOperator rhs = binary.getChild(1);
            if (leftOutputColumns.containsAll(lhs.getUsedColumns()) &&
                    rightOutputColumns.containsAll(rhs.getUsedColumns())) {
                derivePredicates(
                        rewriteByLineage(lhs, leftLineage), rewriteByLineage(rhs, rightLineage),
                        rightBounds, columnRefFactory, leftExistingPredicates, leftPredicates);
                derivePredicates(
                        rewriteByLineage(rhs, rightLineage), rewriteByLineage(lhs, leftLineage),
                        leftBounds, columnRefFactory, rightExistingPredicates, rightPredicates);
            } else if (leftOutputColumns.containsAll(rhs.getUsedColumns()) &&
                    rightOutputColumns.containsAll(lhs.getUsedColumns())) {
                derivePredicates(
                        rewriteByLineage(rhs, leftLineage), rewriteByLineage(lhs, rightLineage),
                        rightBounds, columnRefFactory, leftExistingPredicates, leftPredicates);
                derivePredicates(
                        rewriteByLineage(lhs, rightLineage), rewriteByLineage(rhs, leftLineage),
                        leftBounds, columnRefFactory, rightExistingPredicates, rightPredicates);
            }
        }

        if (leftPredicates.isEmpty() && rightPredicates.isEmpty()) {
            return Collections.emptyList();
        }

        OptExpression newLeftChild = appendPredicates(leftChild, leftPredicates);
        OptExpression newRightChild = appendPredicates(rightChild, rightPredicates);
        return Lists.newArrayList(OptExpression.create(join, newLeftChild, newRightChild));
    }

    private Map<ColumnRefOperator, ScalarOperator> getLineage(OptExpression input, ColumnRefFactory columnRefFactory) {
        LogicalOperator operator = input.getOp().cast();
        if (operator instanceof LogicalProjectOperator) {
            return ((LogicalProjectOperator) operator).getColumnRefMap();
        }
        return operator.getLineage(columnRefFactory, new ExpressionContext(input));
    }

    private ScalarOperator rewriteByLineage(ScalarOperator scalarOperator,
                                            Map<ColumnRefOperator, ScalarOperator> lineage) {
        return new ReplaceColumnRefRewriter(lineage, true).rewrite(scalarOperator);
    }

    private void derivePredicates(ScalarOperator targetExpression,
                                  ScalarOperator sourceExpression,
                                  Map<ColumnRefOperator, ColumnBounds> sourceBounds,
                                  ColumnRefFactory columnRefFactory,
                                  Set<String> existingPredicates,
                                  List<ScalarOperator> outputPredicates) {
        Optional<ColumnRefOperator> targetColumn = extractTargetColumn(targetExpression);
        if (!targetColumn.isPresent()) {
            return;
        }

        Optional<ColumnRefOperator> sourceColumn = extractSingleColumn(sourceExpression, columnRefFactory);
        if (!sourceColumn.isPresent()) {
            return;
        }
        ColumnBounds bounds = sourceBounds.get(sourceColumn.get());
        if (bounds == null || !bounds.hasLowerAndUpper()) {
            return;
        }
        if (!OperatorFunctionChecker.onlyContainMonotonicFunctions(sourceExpression).first) {
            return;
        }

        Optional<ConstantOperator> lower = evaluateExpressionAtEndpoint(sourceExpression, sourceColumn.get(), bounds.lower);
        Optional<ConstantOperator> upper = evaluateExpressionAtEndpoint(sourceExpression, sourceColumn.get(), bounds.upper);
        if (!lower.isPresent() || !upper.isPresent()) {
            return;
        }

        Optional<ConstantOperator> targetLower = lower.get().castTo(targetColumn.get().getType());
        Optional<ConstantOperator> targetUpper = upper.get().castTo(targetColumn.get().getType());
        if (!targetLower.isPresent() || !targetUpper.isPresent() ||
                targetLower.get().isNull() || targetUpper.get().isNull()) {
            return;
        }

        ConstantOperator min = targetLower.get();
        ConstantOperator max = targetUpper.get();
        if (min.compareTo(max) > 0) {
            min = targetUpper.get();
            max = targetLower.get();
        }

        // prevent generation gt/lt for existing predicates
        // fix: PREDICATES: 1: v1 >= 1, 1: v1 <= 1, 1: v1 = 1
        if (min.compareTo(max) == 0 && hasExistingEqualityPredicate(targetColumn.get(), min, existingPredicates)) {
            return;
        }

        addDerivedPredicate(new BinaryPredicateOperator(BinaryType.GE, targetColumn.get(), min),
                existingPredicates, outputPredicates);
        addDerivedPredicate(new BinaryPredicateOperator(BinaryType.LE, targetColumn.get(), max),
                existingPredicates, outputPredicates);
    }

    private Optional<ColumnRefOperator> extractTargetColumn(ScalarOperator expression) {
        if (expression instanceof ColumnRefOperator) {
            return Optional.of((ColumnRefOperator) expression);
        }
        if (expression instanceof CastOperator && expression.getChild(0) instanceof ColumnRefOperator &&
                isSafeTargetCast(expression.getChild(0).getType(), expression.getType())) {
            return Optional.of((ColumnRefOperator) expression.getChild(0));
        }
        return Optional.empty();
    }

    private boolean isSafeTargetCast(Type sourceType, Type targetType) {
        if (sourceType.equals(targetType)) {
            return true;
        }
        return targetType.isStringType() && (sourceType.isFixedPointType() || sourceType.isStringType());
    }

    private Optional<ColumnRefOperator> extractSingleColumn(ScalarOperator expression,
                                                            ColumnRefFactory columnRefFactory) {
        ColumnRefSet usedColumns = expression.getUsedColumns();
        if (usedColumns.cardinality() != 1) {
            return Optional.empty();
        }
        return Optional.of(columnRefFactory.getColumnRef(usedColumns.getFirstId()));
    }

    private Optional<ConstantOperator> evaluateExpressionAtEndpoint(ScalarOperator expression,
                                                                    ColumnRefOperator sourceColumn,
                                                                    ConstantOperator endpoint) {
        Map<ColumnRefOperator, ScalarOperator> replaceMap = Maps.newHashMap();
        replaceMap.put(sourceColumn, endpoint);
        ScalarOperator replaced = new ReplaceColumnRefRewriter(replaceMap).rewrite(expression);
        ScalarOperator folded = new ScalarOperatorRewriter().rewrite(replaced, CONSTANT_REWRITE_RULES);
        if (folded instanceof ConstantOperator && !((ConstantOperator) folded).isNull()) {
            return Optional.of((ConstantOperator) folded);
        }
        return Optional.empty();
    }

    private boolean hasExistingEqualityPredicate(ColumnRefOperator column,
                                                 ConstantOperator constant,
                                                 Set<String> existingPredicates) {
        return existingPredicates.contains(new BinaryPredicateOperator(BinaryType.EQ, column, constant).toString());
    }

    private void addDerivedPredicate(BinaryPredicateOperator predicate,
                                     Set<String> existingPredicates,
                                     List<ScalarOperator> outputPredicates) {
        markAsRedundantRangePredicate(predicate);
        String key = predicate.toString();
        if (existingPredicates.add(key)) {
            outputPredicates.add(predicate);
        }
    }

    private void markAsRedundantRangePredicate(ScalarOperator predicate) {
        predicate.setFromPredicateRangeDerive(true);
        predicate.setNotEvalEstimate(true);
        predicate.setRedundant(true);
    }

    private OptExpression appendPredicates(OptExpression input, List<ScalarOperator> predicates) {
        if (predicates.isEmpty()) {
            return input;
        }

        if (input.getOp() instanceof LogicalProjectOperator && input.arity() == 1 &&
                input.inputAt(0).getOutputColumns().containsAll(getUsedColumns(predicates))) {
            return OptExpression.create(input.getOp(), appendPredicates(input.inputAt(0), predicates));
        }

        if (input.getOp() instanceof LogicalJoinOperator && input.arity() == 2) {
            LogicalJoinOperator join = input.getOp().cast();
            ColumnRefSet usedColumns = getUsedColumns(predicates);
            if (join.getJoinType().isInnerJoin() && input.inputAt(0).getOutputColumns().containsAll(usedColumns)) {
                return OptExpression.create(input.getOp(), appendPredicates(input.inputAt(0), predicates),
                        input.inputAt(1));
            }
            if (join.getJoinType().isInnerJoin() && input.inputAt(1).getOutputColumns().containsAll(usedColumns)) {
                return OptExpression.create(input.getOp(), input.inputAt(0),
                        appendPredicates(input.inputAt(1), predicates));
            }
        }

        LogicalOperator operator = input.getOp().cast();
        List<ScalarOperator> compoundPredicates = Lists.newArrayList(predicates);
        compoundPredicates.add(operator.getPredicate());
        Operator newOperator = OperatorBuilderFactory.build(operator).withOperator(operator)
                .setPredicate(Utils.compoundAnd(compoundPredicates))
                .build();
        OptExpression rewritten = OptExpression.create(newOperator, input.getInputs());
        resetPartitionPruned(rewritten);
        return rewritten;
    }

    private ColumnRefSet getUsedColumns(List<ScalarOperator> predicates) {
        ColumnRefSet usedColumns = new ColumnRefSet();
        predicates.forEach(predicate -> usedColumns.union(predicate.getUsedColumns()));
        return usedColumns;
    }

    private void resetPartitionPruned(OptExpression input) {
        if (input.getOp() instanceof LogicalScanOperator) {
            input.getOp().resetOpRuleBit(OP_PARTITION_PRUNED);
        }
        for (OptExpression child : input.getInputs()) {
            resetPartitionPruned(child);
        }
    }

    private Map<ColumnRefOperator, ColumnBounds> collectColumnBounds(OptExpression input,
                                                                     ColumnRefFactory columnRefFactory) {
        Map<ColumnRefOperator, ColumnBounds> result = Maps.newHashMap();
        collectColumnBounds(input, columnRefFactory, result);
        return result;
    }

    private void collectColumnBounds(OptExpression input,
                                     ColumnRefFactory columnRefFactory,
                                     Map<ColumnRefOperator, ColumnBounds> bounds) {
        for (ScalarOperator conjunct : Utils.extractConjuncts(input.getOp().getPredicate())) {
            collectColumnBound(conjunct, columnRefFactory, bounds);
        }
        for (OptExpression child : input.getInputs()) {
            collectColumnBounds(child, columnRefFactory, bounds);
        }
    }

    private void collectColumnBound(ScalarOperator predicate,
                                    ColumnRefFactory columnRefFactory,
                                    Map<ColumnRefOperator, ColumnBounds> bounds) {
        if (!(predicate instanceof BinaryPredicateOperator)) {
            return;
        }

        BinaryPredicateOperator binaryPredicate = (BinaryPredicateOperator) predicate;
        if (!binaryPredicate.getBinaryType().isEqualOrRange()) {
            return;
        }

        BinaryType type = binaryPredicate.getBinaryType();
        ScalarOperator columnOperator = binaryPredicate.getChild(0);
        ScalarOperator constantOperator = binaryPredicate.getChild(1);
        if (!(columnOperator instanceof ColumnRefOperator) && constantOperator instanceof ColumnRefOperator) {
            type = type.commutative();
            columnOperator = binaryPredicate.getChild(1);
            constantOperator = binaryPredicate.getChild(0);
        }
        if (!(columnOperator instanceof ColumnRefOperator) || !constantOperator.getUsedColumns().isEmpty()) {
            return;
        }

        ColumnRefOperator column = (ColumnRefOperator) columnOperator;
        Optional<ConstantOperator> constant = evaluateConstant(constantOperator);
        if (!constant.isPresent() || constant.get().isNull()) {
            return;
        }
        Optional<ConstantOperator> typedConstant = constant.get().castTo(column.getType());
        if (!typedConstant.isPresent() || typedConstant.get().isNull()) {
            return;
        }

        ColumnBounds columnBounds = bounds.computeIfAbsent(column, ignored -> new ColumnBounds());
        if (BinaryType.EQ.equals(type)) {
            columnBounds.updateLower(typedConstant.get());
            columnBounds.updateUpper(typedConstant.get());
        } else if (BinaryType.GE.equals(type) || BinaryType.GT.equals(type)) {
            columnBounds.updateLower(typedConstant.get());
        } else if (BinaryType.LE.equals(type) || BinaryType.LT.equals(type)) {
            columnBounds.updateUpper(typedConstant.get());
        }
    }

    private Optional<ConstantOperator> evaluateConstant(ScalarOperator expression) {
        ScalarOperator folded = new ScalarOperatorRewriter().rewrite(expression, CONSTANT_REWRITE_RULES);
        if (folded instanceof ConstantOperator) {
            return Optional.of((ConstantOperator) folded);
        }
        return Optional.empty();
    }

    private Set<String> collectPredicateKeys(OptExpression input) {
        Set<String> result = Sets.newHashSet();
        collectPredicateKeys(input, result);
        return result;
    }

    private void collectPredicateKeys(OptExpression input, Set<String> result) {
        for (ScalarOperator conjunct : Utils.extractConjuncts(input.getOp().getPredicate())) {
            result.add(conjunct.toString());
        }
        for (OptExpression child : input.getInputs()) {
            collectPredicateKeys(child, result);
        }
    }

    private static class ColumnBounds {
        private ConstantOperator lower;
        private ConstantOperator upper;

        private boolean hasLowerAndUpper() {
            return lower != null && upper != null;
        }

        private void updateLower(ConstantOperator candidate) {
            if (lower == null || lower.compareTo(candidate) < 0) {
                lower = candidate;
            }
        }

        private void updateUpper(ConstantOperator candidate) {
            if (upper == null || upper.compareTo(candidate) > 0) {
                upper = candidate;
            }
        }
    }
}
