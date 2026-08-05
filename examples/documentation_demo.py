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

"""Demo of Victor's Documentation Generation Tool.

Demonstrates automated documentation generation:
- Generate docstrings for functions and classes
- Create API documentation
- Analyze documentation coverage

Usage:
    python examples/documentation_demo.py
"""

import asyncio
import tempfile
from pathlib import Path

from victor.tools.documentation_tool import docs, docs_coverage


def setup_demo_file(temp_dir: Path, filename: str, content: str) -> Path:
    """Create a demo file."""
    file_path = temp_dir / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return file_path


async def demo_docstrings():
    """Demo generating docstrings."""
    print("\n\n📝 Generate Docstrings Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create code without docstrings
        demo_code = """
def calculate_total(items, tax_rate=0.08):
    if not items:
        return 0
    subtotal = sum(item['price'] for item in items)
    tax = subtotal * tax_rate
    return subtotal + tax


def validate_email(email):
    if not email or '@' not in email:
        raise ValueError("Invalid email")
    return email.lower()


class ShoppingCart:
    def __init__(self):
        self.items = []
        self.discount_code = None

    def add_item(self, product_id, name, price, quantity=1):
        if price < 0:
            raise ValueError("Price cannot be negative")
        self.items.append({
            'product_id': product_id,
            'name': name,
            'price': price,
            'quantity': quantity,
        })

    def get_total(self):
        subtotal = sum(item['price'] * item['quantity'] for item in self.items)
        if self.discount_code == 'SAVE10':
            subtotal *= 0.9
        return round(subtotal, 2)
"""
        file_path = setup_demo_file(temp_path, "cart.py", demo_code.strip())

        print("\n1️⃣ Original code (no docstrings):")
        print(demo_code[:400] + "...")

        print("\n2️⃣ Generate docstrings...")
        result = await docs(
            path=str(file_path),
            doc_types=["docstrings"],
            format="google",
        )

        if result["success"]:
            print(result.get("formatted_report", ""))

        print("\n3️⃣ Updated code with docstrings:")
        updated_code = file_path.read_text()
        print(updated_code[:800] + "...")


async def demo_generate_api_docs():
    """Demo generating API documentation."""
    print("\n\n📚 Generate API Documentation Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create API code with docstrings
        demo_code = '''
"""API module for user management."""


def create_user(username, email, password):
    """Create a new user account.

    Args:
        username: Unique username for the account
        email: User email address
        password: Account password

    Returns:
        Created user object with ID

    Raises:
        ValueError: If username already exists
    """
    return {"id": 1, "username": username, "email": email}


def get_user(user_id):
    """Get user by ID.

    Args:
        user_id: The user identifier

    Returns:
        User object if found, None otherwise
    """
    return {"id": user_id, "username": "testuser"}


class UserManager:
    """Manage user accounts and authentication.

    This class provides methods for creating, updating,
    and authenticating user accounts.

    Attributes:
        db: Database connection
        users: List of active users
    """

    def __init__(self, db):
        """Initialize user manager.

        Args:
            db: Database connection instance
        """
        self.db = db
        self.users = []

    def authenticate(self, username, password):
        """Authenticate user credentials.

        Args:
            username: Username to authenticate
            password: Password to verify

        Returns:
            True if credentials are valid, False otherwise
        """
        return True

    def update_profile(self, user_id, **kwargs):
        """Update user profile.

        Args:
            user_id: ID of user to update
            **kwargs: Profile fields to update

        Returns:
            Updated user object
        """
        return {"id": user_id, **kwargs}
'''
        file_path = setup_demo_file(temp_path, "api.py", demo_code.strip())

        print("\n1️⃣ Generate API documentation from code...")
        result = await docs(
            path=str(file_path),
            doc_types=["api"],
            format="markdown",
            output=str(temp_path / "api_docs.md"),
        )

        if result["success"]:
            print(result.get("formatted_report", ""))
            api_result = result.get("results", {}).get("api", {})
            print("\nGenerated documentation preview:")
            print(api_result.get("preview", ""))


async def demo_analyze_coverage():
    """Demo analyzing documentation coverage."""
    print("\n\n📊 Analyze Documentation Coverage Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create partially documented code
        demo_code = '''
"""Module for data processing."""


def process_data(items):
    """Process data items."""
    return [item.upper() for item in items]


def transform_data(data):
    # No docstring - missing!
    return data.lower()


class DataProcessor:
    """Process and transform data."""

    def __init__(self, config):
        """Initialize processor."""
        self.config = config

    def process(self, data):
        # No docstring - missing!
        return data

    def validate(self, data):
        """Validate data."""
        return True


def another_function(x, y):
    # No docstring - missing!
    return x + y
'''
        file_path = setup_demo_file(temp_path, "processor.py", demo_code.strip())

        print("\n1️⃣ Analyze documentation coverage...")
        result = await docs_coverage(path=str(file_path))

        if result["success"]:
            print(result.get("formatted_report", ""))


async def demo_real_world_workflow():
    """Demo a real-world documentation workflow."""
    print("\n\n🎯 Real-World Documentation Workflow Demo")
    print("=" * 70)
    print("\nScenario: Documenting an existing codebase")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create realistic undocumented code
        demo_code = """
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class PaymentProcessor:
    def __init__(self, api_key, sandbox=False):
        self.api_key = api_key
        self.sandbox = sandbox
        self.transactions = []

    def charge(self, amount, currency, customer_id, metadata=None):
        if amount <= 0:
            raise ValueError("Amount must be positive")

        transaction = {
            'amount': amount,
            'currency': currency,
            'customer_id': customer_id,
            'metadata': metadata or {},
            'status': 'pending'
        }

        self.transactions.append(transaction)
        return transaction

    def refund(self, transaction_id, amount=None):
        for txn in self.transactions:
            if txn.get('id') == transaction_id:
                refund_amount = amount or txn['amount']
                txn['status'] = 'refunded'
                return {'refunded': refund_amount}
        return None

    def get_transaction(self, transaction_id):
        for txn in self.transactions:
            if txn.get('id') == transaction_id:
                return txn
        return None


def validate_card(card_number, exp_month, exp_year, cvv):
    if len(card_number) not in (15, 16):
        return False
    if exp_month < 1 or exp_month > 12:
        return False
    if len(str(cvv)) not in (3, 4):
        return False
    return True


def format_currency(amount, currency='USD'):
    symbols = {'USD': '$', 'EUR': '€', 'GBP': '£'}
    symbol = symbols.get(currency, currency)
    return f"{symbol}{amount:.2f}"
"""
        file_path = setup_demo_file(temp_path, "payment.py", demo_code.strip())

        print("\n1️⃣ STEP 1: Analyze current documentation coverage...")
        result = await docs_coverage(path=str(file_path))
        if result["success"]:
            print("✓ Coverage analyzed")
            report = result.get("formatted_report", "")
            print(report[:400] + "..." if len(report) > 400 else report)

        print("\n\n2️⃣ STEP 2: Generate missing docstrings...")
        result = await docs(
            path=str(file_path),
            doc_types=["docstrings"],
        )
        if result["success"]:
            print("✓ Docstrings generated")
            print(result.get("formatted_report", ""))

        print("\n\n3️⃣ STEP 3: Generate API documentation...")
        result = await docs(
            path=str(file_path),
            doc_types=["api"],
            output=str(temp_path / "payment_api.md"),
        )
        if result["success"]:
            print("✓ API docs generated")

        print("\n\n4️⃣ STEP 4: Verify final coverage...")
        result = await docs_coverage(path=str(file_path))
        if result["success"]:
            print("✓ Final coverage verified")
            report = result.get("formatted_report", "")
            print(report[:300] + "..." if len(report) > 300 else report)

        print("\n\n📊 Documentation Workflow Complete!")
        print("\nWhat we accomplished:")
        print("  • Analyzed initial coverage (likely low)")
        print("  • Generated comprehensive docstrings")
        print("  • Created API documentation")
        print("  • Verified improved coverage")
        print("")
        print("Files created:")
        print("  • payment.py (with complete docstrings)")
        print("  • payment_api.md (API documentation)")


async def demo_documentation_benefits():
    """Demo documentation benefits."""
    print("\n\n⭐ Documentation Benefits Demo")
    print("=" * 70)

    print("\n✨ Benefits of Good Documentation:")
    print("")

    print("1️⃣ For Developers:")
    print("  • Faster onboarding for new team members")
    print("  • Clear API contracts and expectations")
    print("  • Easier code maintenance and updates")
    print("  • Better IDE autocomplete and hints")
    print("")

    print("2️⃣ For Users:")
    print("  • Clear usage examples and patterns")
    print("  • Understanding parameters and returns")
    print("  • Knowledge of error conditions")
    print("  • Complete API reference")
    print("")

    print("3️⃣ For Projects:")
    print("  • Professional appearance")
    print("  • Increased adoption")
    print("  • Reduced support burden")
    print("  • Better collaboration")
    print("")

    print("4️⃣ Documentation Standards:")
    print("  • Google Style: Clear, readable, popular")
    print("  • NumPy Style: Scientific computing standard")
    print("  • Sphinx: Auto-generated HTML docs")
    print("  • Markdown: Simple, universal format")
    print("")

    print("5️⃣ Best Practices:")
    print("  • Document all public APIs")
    print("  • Include examples in docstrings")
    print("  • Explain parameters and returns")
    print("  • Note exceptions and edge cases")
    print("  • Keep docs up-to-date with code")
    print("")


async def main():
    """Run all documentation demos."""
    print("🎯 Victor Documentation Tool Demo")
    print("=" * 70)
    print("\nDemonstrating automated documentation generation\n")

    # Run demos
    await demo_docstrings()
    await demo_generate_api_docs()
    await demo_analyze_coverage()
    await demo_real_world_workflow()
    await demo_documentation_benefits()

    print("\n\n✨ Demo Complete!")
    print("\nVictor's Documentation Tool provides:")
    print("  • Automated docstring generation")
    print("  • API documentation creation")
    print("  • Documentation coverage analysis")
    print("  • Multiple documentation formats")
    print("")
    print("Supported formats:")
    print("  • Google Style docstrings")
    print("  • NumPy Style docstrings")
    print("  • Markdown documentation")
    print("  • reStructuredText (RST)")
    print("")
    print("Perfect for:")
    print("  • Documenting existing codebases")
    print("  • Maintaining documentation standards")
    print("  • Generating API references")
    print("  • Improving code discoverability")
    print("  • Professional project presentation")
    print("")
    print("Ready to document your code!")


if __name__ == "__main__":
    asyncio.run(main())
