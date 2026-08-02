# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Teams integration for Data Analysis vertical.

This package provides team specifications for common data analysis tasks with
rich persona attributes for natural agent characterization.

Example:
    from victor_dataanalysis.teams import (
        get_team_for_task,
        DATA_ANALYSIS_TEAM_SPECS,
    )

    # Get team for a task type
    team_spec = get_team_for_task("eda")
    print(f"Team: {team_spec.name}")
    print(f"Members: {len(team_spec.members)}")

Teams are auto-registered with the global TeamSpecRegistry on import,
enabling cross-vertical team discovery via:
    from victor_contracts.team_schema import get_runtime_team_registry as get_team_registry
    registry = get_team_registry()
    data_analysis_teams = registry.find_by_vertical("data_analysis")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from victor_contracts import TeamFormation, TeamMemberSpec


@dataclass
class DataAnalysisRoleConfig:
    """Configuration for a Data Analysis-specific role.

    Attributes:
        base_role: Base agent role (researcher, analyst, executor, reviewer)
        tools: Tools available to this role
        tool_budget: Default tool budget
        description: Role description
    """

    base_role: str
    tools: List[str]
    tool_budget: int
    description: str = ""


# Data Analysis-specific roles with tool allocations
DATA_ANALYSIS_ROLES: Dict[str, DataAnalysisRoleConfig] = {
    "data_loader": DataAnalysisRoleConfig(
        base_role="researcher",
        tools=["read_file", "bash", "code_search"],
        tool_budget=15,
        description="Loads and validates data from various sources",
    ),
    "data_profiler": DataAnalysisRoleConfig(
        base_role="analyst",
        tools=["bash", "read_file", "write_file"],
        tool_budget=25,
        description="Profiles data and generates summary statistics",
    ),
    "data_cleaner": DataAnalysisRoleConfig(
        base_role="executor",
        tools=["bash", "write_file", "edit_files", "read_file"],
        tool_budget=30,
        description="Cleans and transforms data",
    ),
    "visualizer": DataAnalysisRoleConfig(
        base_role="executor",
        tools=["bash", "write_file", "read_file"],
        tool_budget=25,
        description="Creates charts and visualizations",
    ),
    "statistical_analyst": DataAnalysisRoleConfig(
        base_role="executor",
        tools=["bash", "read_file", "write_file"],
        tool_budget=35,
        description="Runs statistical tests and analyses",
    ),
    "model_trainer": DataAnalysisRoleConfig(
        base_role="executor",
        tools=["bash", "read_file", "write_file"],
        tool_budget=40,
        description="Trains and tunes machine learning models",
    ),
}


@dataclass
class DataAnalysisTeamSpec:
    """Specification for a data analysis team.

    Attributes:
        name: Team name
        description: Team description
        formation: How agents are organized
        members: Team member specifications
        total_tool_budget: Total tool budget for the team
        max_iterations: Maximum iterations
    """

    name: str
    description: str
    formation: TeamFormation
    members: List[TeamMemberSpec]
    total_tool_budget: int = 100
    max_iterations: int = 50


# Pre-defined team specifications with rich personas
DATA_ANALYSIS_TEAM_SPECS: Dict[str, DataAnalysisTeamSpec] = {
    "eda_team": DataAnalysisTeamSpec(
        name="Exploratory Data Analysis Team",
        description="Comprehensive data exploration with profiling and visualization",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Load data and understand its structure and schema",
                name="Data Loader",
                tool_budget=15,
                backstory=(
                    "You are a data engineer who has worked with every data format imaginable: "
                    "CSV, JSON, Parquet, databases, APIs, and legacy systems. You know the quirks "
                    "of each format and can detect encoding issues, delimiter problems, and "
                    "malformed records. You load data defensively, validating types and handling "
                    "edge cases. You document what you find about the data's structure."
                ),
                expertise=["data formats", "schema inference", "data validation"],
                personality="meticulous and defensive; expects data to be messy",
                memory=True,  # Share schema with profiler
            ),
            TeamMemberSpec(
                role="analyst",
                goal="Generate comprehensive data profiles and summary statistics",
                name="Data Profiler",
                tool_budget=25,
                backstory=(
                    "You are a statistical profiling expert who can summarize any dataset in "
                    "minutes. You compute not just means and medians, but distributions, "
                    "correlations, missing value patterns, and outliers. You know that summary "
                    "statistics can lie, so you always look at the shape of the data. You flag "
                    "potential data quality issues and anomalies."
                ),
                expertise=[
                    "descriptive statistics",
                    "distribution analysis",
                    "data quality",
                ],
                personality="thorough and curious; wants to understand every column",
                memory=True,  # Share profile with visualizer
            ),
            TeamMemberSpec(
                role="executor",
                goal="Create insightful visualizations that reveal patterns",
                name="Data Visualizer",
                tool_budget=25,
                backstory=(
                    "You are a data visualization artist who follows Tufte's principles. You "
                    "create charts that reveal rather than obscure. You choose the right chart "
                    "type for each variable: histograms for distributions, scatter plots for "
                    "relationships, bar charts for categories. You label axes, add context, "
                    "and make visualizations that can stand alone."
                ),
                expertise=["matplotlib", "seaborn", "plotly", "visualization design"],
                personality="visual thinker; believes a good chart is worth a thousand numbers",
            ),
            TeamMemberSpec(
                role="writer",
                goal="Synthesize insights into an EDA report",
                name="Insight Synthesizer",
                tool_budget=15,
                backstory=(
                    "You are a data storyteller who turns exploratory analysis into actionable "
                    "insights. You know which findings matter and which are noise. You structure "
                    "EDA reports with executive summaries, key findings, and recommendations. "
                    "You connect the dots between different variables and suggest next steps."
                ),
                expertise=["data storytelling", "insight synthesis", "report writing"],
                personality="insightful and actionable; focuses on 'so what?'",
            ),
        ],
        total_tool_budget=80,
    ),
    "cleaning_team": DataAnalysisTeamSpec(
        name="Data Cleaning Team",
        description="Systematic data quality assessment and cleaning",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="analyst",
                goal="Assess data quality issues and document problems",
                name="Quality Assessor",
                tool_budget=15,
                backstory=(
                    "You are a data quality detective who finds every issue. You check for "
                    "missing values, duplicates, inconsistent formats, invalid values, and "
                    "logical contradictions. You categorize issues by severity and impact. "
                    "You know that data quality issues often hide in edge cases."
                ),
                expertise=[
                    "data quality assessment",
                    "anomaly detection",
                    "data profiling",
                ],
                personality="suspicious and thorough; assumes data is guilty until proven innocent",
                memory=True,  # Share issues with planner
            ),
            TeamMemberSpec(
                role="planner",
                goal="Design a cleaning strategy that preserves data integrity",
                name="Cleaning Strategist",
                tool_budget=10,
                backstory=(
                    "You are a data quality strategist who balances thoroughness with pragmatism. "
                    "You prioritize cleaning steps by impact and risk. You know when to impute, "
                    "when to drop, and when to flag for manual review. You design cleaning "
                    "pipelines that are reproducible and auditable."
                ),
                expertise=[
                    "data cleaning strategy",
                    "missing value handling",
                    "reproducibility",
                ],
                personality="strategic and pragmatic; makes defensible trade-offs",
                memory=True,  # Share plan with cleaner
            ),
            TeamMemberSpec(
                role="executor",
                goal="Execute cleaning transformations accurately",
                name="Data Cleaner",
                tool_budget=30,
                backstory=(
                    "You are a data wrangling expert who writes clean, efficient pandas code. "
                    "You apply transformations systematically, handling edge cases and logging "
                    "what you change. You use vectorized operations for performance. You create "
                    "reusable cleaning functions that can be applied to future data."
                ),
                expertise=["pandas", "data wrangling", "ETL", "Python"],
                personality="precise and efficient; writes code that explains itself",
                cache=True,  # Cache file reads
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Validate that cleaned data meets quality standards",
                name="Quality Validator",
                tool_budget=15,
                backstory=(
                    "You are a quality gatekeeper who ensures cleaning was successful. You run "
                    "validation checks, compare before/after statistics, and verify that no "
                    "good data was lost. You check for unintended consequences of cleaning. "
                    "You only approve data that meets defined quality standards."
                ),
                expertise=["data validation", "quality metrics", "testing"],
                personality="skeptical and careful; doesn't trust without verification",
            ),
        ],
        total_tool_budget=70,
    ),
    "statistics_team": DataAnalysisTeamSpec(
        name="Statistical Analysis Team",
        description="Rigorous hypothesis testing and statistical modeling",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Formulate hypotheses and design the analysis",
                name="Research Designer",
                tool_budget=15,
                backstory=(
                    "You are a research methodologist who designs rigorous statistical studies. "
                    "You translate business questions into testable hypotheses. You choose "
                    "appropriate statistical methods based on data characteristics and research "
                    "goals. You consider sample size, power, and multiple testing issues."
                ),
                expertise=[
                    "research design",
                    "hypothesis formulation",
                    "statistical methods",
                ],
                personality="rigorous and thoughtful; designs for validity",
                memory=True,  # Share design with analyst
            ),
            TeamMemberSpec(
                role="executor",
                goal="Prepare data and execute statistical tests",
                name="Statistical Analyst",
                tool_budget=35,
                backstory=(
                    "You are a statistical analyst who can run any test correctly. You verify "
                    "assumptions before applying tests. You use scipy, statsmodels, and pingouin "
                    "appropriately. You compute effect sizes, not just p-values. You handle "
                    "multiple comparisons and report results completely."
                ),
                expertise=["scipy", "statsmodels", "hypothesis testing", "regression"],
                personality="methodical and correct; follows statistical best practices",
            ),
            TeamMemberSpec(
                role="analyst",
                goal="Interpret results and assess practical significance",
                name="Result Interpreter",
                tool_budget=20,
                backstory=(
                    "You are a statistical interpreter who bridges numbers and meaning. You "
                    "distinguish between statistical and practical significance. You explain "
                    "confidence intervals, effect sizes, and uncertainty. You put results in "
                    "context and discuss limitations honestly."
                ),
                expertise=[
                    "statistical interpretation",
                    "effect size",
                    "scientific communication",
                ],
                personality="nuanced and honest; never oversells findings",
            ),
            TeamMemberSpec(
                role="writer",
                goal="Write a clear statistical report with conclusions",
                name="Statistical Writer",
                tool_budget=15,
                backstory=(
                    "You are a statistical writer who makes complex analyses accessible. You "
                    "structure reports with methods, results, and discussion sections. You "
                    "include appropriate visualizations. You state conclusions clearly and "
                    "acknowledge limitations."
                ),
                expertise=[
                    "statistical writing",
                    "data visualization",
                    "scientific communication",
                ],
                personality="clear and precise; writes for the target audience",
            ),
        ],
        total_tool_budget=85,
    ),
    "ml_team": DataAnalysisTeamSpec(
        name="Machine Learning Team",
        description="End-to-end ML pipeline from problem definition to model evaluation",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Define the ML problem and success metrics",
                name="Problem Definer",
                tool_budget=10,
                backstory=(
                    "You are an ML strategist who ensures we solve the right problem. You "
                    "translate business goals into ML objectives. You define appropriate "
                    "metrics that align with business value. You identify potential pitfalls "
                    "like data leakage and selection bias early."
                ),
                expertise=[
                    "ML problem framing",
                    "metrics selection",
                    "business alignment",
                ],
                personality="strategic and questioning; asks 'why' before 'how'",
                memory=True,  # Share problem definition
            ),
            TeamMemberSpec(
                role="executor",
                goal="Engineer predictive features from raw data",
                name="Feature Engineer",
                tool_budget=25,
                backstory=(
                    "You are a feature engineering expert who creates predictive signals. You "
                    "understand domain knowledge and encode it into features. You handle "
                    "categorical variables, dates, text, and interactions. You validate that "
                    "features are available at prediction time."
                ),
                expertise=[
                    "feature engineering",
                    "sklearn pipelines",
                    "domain encoding",
                ],
                personality="creative and practical; turns data into signal",
                memory=True,  # Share features with trainer
            ),
            TeamMemberSpec(
                role="executor",
                goal="Train, tune, and validate ML models",
                name="Model Trainer",
                tool_budget=35,
                backstory=(
                    "You are an ML practitioner who trains models that generalize. You use "
                    "proper cross-validation, avoid data leakage, and tune hyperparameters "
                    "systematically. You try appropriate algorithms for the problem type. "
                    "You track experiments and ensure reproducibility."
                ),
                expertise=[
                    "scikit-learn",
                    "XGBoost",
                    "cross-validation",
                    "hyperparameter tuning",
                ],
                personality="systematic and experimental; treats ML as science",
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Evaluate models fairly and recommend the best one",
                name="Model Evaluator",
                tool_budget=20,
                backstory=(
                    "You are an ML evaluator who assesses models beyond accuracy. You compute "
                    "multiple metrics, analyze errors, and test for fairness. You validate on "
                    "held-out data and check for overfitting. You document model limitations "
                    "and recommend the best model for deployment."
                ),
                expertise=["model evaluation", "fairness metrics", "error analysis"],
                personality="critical and fair; judges models by their weaknesses",
            ),
        ],
        total_tool_budget=90,
    ),
    "visualization_team": DataAnalysisTeamSpec(
        name="Visualization Team",
        description="Create impactful charts and dashboards",
        formation=TeamFormation.SEQUENTIAL,
        members=[
            TeamMemberSpec(
                role="analyst",
                goal="Identify the key stories that visualizations should tell",
                name="Story Analyst",
                tool_budget=15,
                backstory=(
                    "You are a data story designer who identifies what matters. You understand "
                    "the audience and their questions. You identify the key messages that "
                    "visualizations should convey. You prioritize impact over comprehensiveness."
                ),
                expertise=[
                    "data storytelling",
                    "audience analysis",
                    "visual narratives",
                ],
                personality="strategic and focused; less is more",
                memory=True,  # Share story with creator
            ),
            TeamMemberSpec(
                role="executor",
                goal="Create clear, beautiful, and accurate visualizations",
                name="Visualization Creator",
                tool_budget=35,
                backstory=(
                    "You are a visualization craftsman who creates charts that inform and "
                    "inspire. You follow best practices: clear labels, appropriate scales, "
                    "accessible colors, minimal chartjunk. You use matplotlib, seaborn, or "
                    "plotly as appropriate for the use case."
                ),
                expertise=["matplotlib", "seaborn", "plotly", "design principles"],
                personality="artistic and precise; cares about every pixel",
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Ensure visualizations are accurate and effective",
                name="Visualization Reviewer",
                tool_budget=15,
                backstory=(
                    "You are a visualization critic who ensures quality. You check that charts "
                    "don't mislead, axes are appropriate, and messages are clear. You test "
                    "accessibility and suggest improvements. You only approve visualizations "
                    "that meet professional standards."
                ),
                expertise=["visualization review", "accessibility", "chart accuracy"],
                personality="critical and constructive; makes good charts great",
            ),
        ],
        total_tool_budget=65,
    ),
    "reporting_team": DataAnalysisTeamSpec(
        name="Reporting Team",
        description="Create comprehensive analysis reports and presentations",
        formation=TeamFormation.HIERARCHICAL,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Gather and organize analysis outputs",
                name="Content Gatherer",
                tool_budget=15,
                backstory=(
                    "You are an organized analyst who collects and structures all analysis "
                    "outputs. You gather charts, tables, statistics, and insights. You organize "
                    "them logically and identify gaps. You ensure nothing important is missed."
                ),
                expertise=[
                    "content organization",
                    "information synthesis",
                    "documentation",
                ],
                personality="organized and thorough; creates structure from chaos",
                memory=True,
            ),
            TeamMemberSpec(
                role="writer",
                goal="Write a compelling analysis report",
                name="Report Writer",
                tool_budget=25,
                is_manager=True,  # Coordinates the report
                backstory=(
                    "You are a technical writer who creates reports that drive decisions. You "
                    "structure reports with executive summaries, methodology, findings, and "
                    "recommendations. You write for your audience and use visualizations "
                    "effectively. Your reports are polished and professional."
                ),
                expertise=[
                    "technical writing",
                    "report structure",
                    "executive communication",
                ],
                personality="clear and persuasive; writes for impact",
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Review report for accuracy and clarity",
                name="Report Reviewer",
                tool_budget=10,
                backstory=(
                    "You are an editor who ensures reports are accurate and clear. You check "
                    "that claims are supported, numbers are correct, and writing is concise. "
                    "You improve readability and catch errors. Your review is the final gate."
                ),
                expertise=["editing", "fact-checking", "clarity"],
                personality="meticulous and helpful; makes reports shine",
            ),
        ],
        total_tool_budget=50,
    ),
}


def get_team_for_task(task_type: str) -> Optional[DataAnalysisTeamSpec]:
    """Get appropriate team specification for task type.

    Args:
        task_type: Type of task (eda, clean, statistics, ml, etc.)

    Returns:
        DataAnalysisTeamSpec or None if no matching team
    """
    mapping = {
        # EDA tasks
        "eda": "eda_team",
        "exploration": "eda_team",
        "exploratory": "eda_team",
        "profiling": "eda_team",
        "profile": "eda_team",
        # Cleaning tasks
        "clean": "cleaning_team",
        "cleaning": "cleaning_team",
        "preparation": "cleaning_team",
        "prepare": "cleaning_team",
        "wrangling": "cleaning_team",
        # Statistics tasks
        "statistics": "statistics_team",
        "statistical": "statistics_team",
        "hypothesis": "statistics_team",
        "test": "statistics_team",
        "regression": "statistics_team",
        # ML tasks
        "ml": "ml_team",
        "machine_learning": "ml_team",
        "training": "ml_team",
        "model": "ml_team",
        "prediction": "ml_team",
        # Visualization tasks
        "visualization": "visualization_team",
        "visualize": "visualization_team",
        "chart": "visualization_team",
        "plot": "visualization_team",
        "dashboard": "visualization_team",
        # Reporting tasks
        "report": "reporting_team",
        "reporting": "reporting_team",
        "presentation": "reporting_team",
    }
    spec_name = mapping.get(task_type.lower())
    if spec_name:
        return DATA_ANALYSIS_TEAM_SPECS.get(spec_name)
    return None


def get_role_config(role_name: str) -> Optional[DataAnalysisRoleConfig]:
    """Get configuration for a Data Analysis role.

    Args:
        role_name: Role name

    Returns:
        DataAnalysisRoleConfig or None
    """
    return DATA_ANALYSIS_ROLES.get(role_name.lower())


def list_team_types() -> List[str]:
    """List all available team types.

    Returns:
        List of team type names
    """
    return list(DATA_ANALYSIS_TEAM_SPECS.keys())


def list_roles() -> List[str]:
    """List all available Data Analysis roles.

    Returns:
        List of role names
    """
    return list(DATA_ANALYSIS_ROLES.keys())


class DataAnalysisTeamSpecProvider:
    """Team specification provider for Data Analysis vertical.

    Implements TeamSpecProviderProtocol interface for consistent
    ISP compliance across all verticals.
    """

    def get_team_specs(self) -> Dict[str, DataAnalysisTeamSpec]:
        """Get all Data Analysis team specifications.

        Returns:
            Dictionary mapping team names to DataAnalysisTeamSpec instances
        """
        return DATA_ANALYSIS_TEAM_SPECS

    def get_team_for_task(self, task_type: str) -> Optional[DataAnalysisTeamSpec]:
        """Get appropriate team for a task type.

        Args:
            task_type: Type of task

        Returns:
            DataAnalysisTeamSpec or None if no matching team
        """
        return get_team_for_task(task_type)

    def list_team_types(self) -> List[str]:
        """List all available team types.

        Returns:
            List of team type names
        """
        return list_team_types()


# Import personas
from victor_dataanalysis.teams.personas import (
    # Framework types (re-exported for convenience)
    FrameworkPersonaTraits,
    FrameworkCommunicationStyle,
    ExpertiseLevel,
    PersonaTemplate,
    # Data analysis-specific types
    ExpertiseCategory,
    CommunicationStyle,
    DecisionStyle,
    DataAnalysisPersonaTraits,
    PersonaTraits,  # Backward compatibility alias
    DataAnalysisPersona,
    # Pre-defined personas
    DATA_ANALYSIS_PERSONAS,
    # Helper functions
    get_persona,
    get_personas_for_role,
    get_persona_by_expertise,
    apply_persona_to_spec,
    list_personas,
    register_data_analysis_personas,
)

__all__ = [
    # Types
    "DataAnalysisRoleConfig",
    "DataAnalysisTeamSpec",
    # Provider
    "DataAnalysisTeamSpecProvider",
    # Role configurations
    "DATA_ANALYSIS_ROLES",
    # Team specifications
    "DATA_ANALYSIS_TEAM_SPECS",
    # Helper functions
    "get_team_for_task",
    "get_role_config",
    "list_team_types",
    "list_roles",
    # Framework types (re-exported for convenience)
    "FrameworkPersonaTraits",
    "FrameworkCommunicationStyle",
    "ExpertiseLevel",
    "PersonaTemplate",
    # Data analysis-specific types from personas
    "ExpertiseCategory",
    "CommunicationStyle",
    "DecisionStyle",
    "DataAnalysisPersonaTraits",
    "PersonaTraits",  # Backward compatibility alias
    "DataAnalysisPersona",
    # Pre-defined personas
    "DATA_ANALYSIS_PERSONAS",
    # Helper functions from personas
    "get_persona",
    "get_personas_for_role",
    "get_persona_by_expertise",
    "apply_persona_to_spec",
    "list_personas",
    "register_data_analysis_personas",
]

logger = logging.getLogger(__name__)


def register_data_analysis_teams() -> int:
    """Register data analysis teams with global registry.

    This function is called during vertical integration by the framework's
    step handlers. Import-time auto-registration has been removed to avoid
    load-order coupling and duplicate registration.

    Returns:
        Number of teams registered.
    """
    try:
        from victor_contracts.team_schema import (
            get_runtime_team_registry as get_team_registry,
        )

        registry = get_team_registry()
        count = registry.register_from_vertical("data_analysis", DATA_ANALYSIS_TEAM_SPECS)
        logger.debug(f"Registered {count} data analysis teams via framework integration")
        return count
    except Exception as e:
        logger.warning(f"Failed to register data analysis teams: {e}")
        return 0


# NOTE: Import-time auto-registration removed (SOLID compliance)
# Registration now happens during vertical integration via step_handlers.py
# This avoids load-order coupling and duplicate registration issues.
