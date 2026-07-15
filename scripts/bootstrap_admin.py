import argparse
import asyncio
import getpass
import sys
import os

# Ensure the script can import from app when run from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.db.engine import create_database_engine
from app.db.transactions import transactional
from app.models.enums import OrganizationRole
from app.schemas.membership import MembershipCreate
from app.schemas.organization import OrganizationCreate
from app.schemas.user import UserCreate
from app.services.authentication import AuthenticationService
from app.services.membership import MembershipService
from app.services.organization import OrganizationService
from app.services.user import UserService


async def bootstrap_admin():
    parser = argparse.ArgumentParser(description="Bootstrap the initial SchemaLens organization and admin user.")
    parser.add_argument("--org-name", required=True, help="Name of the initial organization")
    parser.add_argument("--org-slug", required=True, help="Slug for the initial organization")
    parser.add_argument("--email", required=True, help="Administrator email address")
    parser.add_argument("--display-name", required=True, help="Administrator display name")
    parser.add_argument("--platform-admin", action="store_true", help="Set this user as a platform admin")
    args = parser.parse_args()

    password = getpass.getpass(prompt="Enter administrator password: ")
    if not password:
        print("Password is required.")
        sys.exit(1)
        
    password_confirm = getpass.getpass(prompt="Confirm administrator password: ")
    if password != password_confirm:
        print("Passwords do not match.")
        sys.exit(1)

    settings = get_settings()
    engine = create_database_engine(settings)
    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False, autocommit=False
    )

    try:
        async with session_factory() as session:
            org_service = OrganizationService(session)
            user_service = UserService(session)
            mem_service = MembershipService(session)
            auth_service = AuthenticationService(session, settings)
            
            async with transactional(session):
                # 1. Create Organization
                org = await org_service.create_organization(
                    OrganizationCreate(name=args.org_name, slug=args.org_slug)
                )

                # 2. Create User
                user = await user_service.create_user(
                    UserCreate(email=args.email, display_name=args.display_name)
                )

                if args.platform_admin:
                    db_user = await user_service.repository.get_by_id(user.id)
                    db_user.is_platform_admin = True
                    await session.flush()

                # 3. Create Membership
                mem = await mem_service.create_membership(
                    org.id, 
                    MembershipCreate(user_id=user.id, role=OrganizationRole.ORGANIZATION_ADMIN)
                )
                
                # 4. Set Password
                await auth_service.set_initial_password(user.id, password, password_confirm)
                
            print("\nBootstrap complete.")
            print(f"Organization ID: {org.id}")
            print(f"User ID: {user.id}")
            print(f"Membership ID: {mem.id}")
            
    except Exception as e:
        print(f"\nBootstrap failed: {e}")
        sys.exit(1)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(bootstrap_admin())
