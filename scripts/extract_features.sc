import io.shiftleft.semanticcpg.language._

val methods = cpg.method
  .filter(m =>
    m.filename != "<empty>" &&
    m.name != "<global>" &&
    !m.name.startsWith("<operator>")
  )

val output = new java.io.PrintWriter(
  "data/devign/cpg_test/features.csv"
)

output.println(
  "function,filename,ast_nodes,cfg_nodes,ast_depth," +
  "branches,loops,assignments,returns,function_calls,operators," +
  "pointer_ops,array_accesses,field_accesses"
)

methods.foreach { m =>

  val functionName = m.name
  val filename = m.filename

  val astNodes = m.ast.size
  val cfgNodes = m.cfgNode.size

  // AST depth
  val astDepth =
    if (m.ast.nonEmpty)
      m.ast.map(_.depth).max
    else
      0

  // Branches
  val branches = m.controlStructure
    .filter(c =>
      c.controlStructureType == "IF" ||
      c.controlStructureType == "SWITCH"
    ).size

  // Loops
  val loops = m.controlStructure
    .filter(c =>
      c.controlStructureType == "FOR" ||
      c.controlStructureType == "WHILE" ||
      c.controlStructureType == "DO"
    ).size

  // Assignments
  val assignments = m.call
    .nameExact(
      "<operator>.assignment",
      "<operator>.assignmentPlus",
      "<operator>.assignmentMinus",
      "<operators>.assignmentAnd",
      "<operators>.assignmentArithmeticShiftRight"
    ).size

  // Returns
  val returns = m.methodReturn.size

  // Actual function calls
  val functionCalls = m.call
    .filter(c =>
      !c.name.startsWith("<operator>") &&
      !c.name.startsWith("<operators>")
    ).size

  // Operators
  val operators = m.call
    .filter(c =>
      c.name.startsWith("<operator>") ||
      c.name.startsWith("<operators>")
    ).size

  // Pointer operations
  val pointerOps = m.call
    .nameExact(
      "<operator>.indirection",
      "<operator>.addressOf"
    ).size

  // Array indexing
  val arrayAccesses = m.call
    .nameExact("<operator>.indirectIndexAccess")
    .size

  // Struct/field access
  val fieldAccesses = m.call
    .nameExact(
      "<operator>.fieldAccess",
      "<operator>.indirectFieldAccess"
    ).size

  output.println(
    s"$functionName,$filename,$astNodes,$cfgNodes,$astDepth," +
    s"$branches,$loops,$assignments,$returns,$functionCalls,$operators," +
    s"$pointerOps,$arrayAccesses,$fieldAccesses"
  )
}

output.close()

println("Feature extraction complete.")
