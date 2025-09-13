# main.py
import argparse
import json
from blockchain import Blockchain, Transaction

# Global variable for our blockchain
blockchain = Blockchain()

def create_wallet():
    """Creates a new wallet. For v0.1, this just prints a message."""
    # In a real implementation, this would generate a private key and public address.
    # For now, we'll use placeholder addresses.
    print("""
    Wallet Created!
    ---------------
    IMPORTANT: This is a simplified version for v0.1.
    In a real wallet, you would generate a cryptographic keypair.

    Your public address: placeholder_address_xyz123
    Your private key: placeholder_private_key_abc456 (KEEP THIS SECRET!)

    Use this address to receive funds and send transactions.
    """)

def get_balance(address):
    """Gets the balance for a given address."""
    balance = blockchain.get_balance_of_address(address)
    print(f"Balance for address '{address}': {balance} RXY")

def send_coins(sender, recipient, amount, fee):
    """Sends coins from one address to another."""
    # TODO: In a real implementation, you would sign the transaction with the sender's private key.
    # For v0.1, we skip the signature validation.
    transaction = Transaction(sender, recipient, amount, fee, signature="unsigned_v0.1")
    success = blockchain.add_transaction(transaction)
    if success:
        print(f"Successfully added transaction to mempool.")
        print(f"{sender} -> {recipient} Amount: {amount} RXY Fee: {fee} RXY")
        print("The transaction will be confirmed once a block is mined.")
    else:
        print("Failed to add transaction.")

def mine_block(miner_address):
    """Mines a new block with the current pending transactions."""
    print(f"Miner address: {miner_address}")
    print("Starting mining process...")
    success = blockchain.mine_pending_transactions(miner_address)
    if success:
        print("Mining successful! A new block has been added to the chain.")
    else:
        print("Mining failed. No transactions to mine.")

def show_chain():
    """Prints the entire blockchain."""
    print("\nRAYONIX BLOCKCHAIN")
    print("==================")
    blockchain.print_chain()

def main():
    parser = argparse.ArgumentParser(description='RAYONIX CLI - Interact with the RAYONIX Blockchain')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Create wallet command
    parser_create = subparsers.add_parser('create-wallet', help='Create a new wallet')

    # Get balance command
    parser_balance = subparsers.add_parser('get-balance', help='Get wallet balance')
    parser_balance.add_argument('address', type=str, help='Wallet address to check balance for')

    # Send coins command
    parser_send = subparsers.add_parser('send', help='Send coins to an address')
    parser_send.add_argument('sender', type=str, help='Sender wallet address')
    parser_send.add_argument('recipient', type=str, help='Recipient wallet address')
    parser_send.add_argument('amount', type=int, help='Amount of RXY to send')
    parser_send.add_argument('--fee', type=int, default=1, help='Transaction fee (default: 1)')

    # Mine command
    parser_mine = subparsers.add_parser('mine', help='Mine a new block')
    parser_mine.add_argument('miner_address', type=str, help='Miner reward address')

    # Show chain command
    parser_chain = subparsers.add_parser('show-chain', help='Show the entire blockchain')

    args = parser.parse_args()

    if args.command == 'create-wallet':
        create_wallet()
    elif args.command == 'get-balance':
        get_balance(args.address)
    elif args.command == 'send':
        send_coins(args.sender, args.recipient, args.amount, args.fee)
    elif args.command == 'mine':
        mine_block(args.miner_address)
    elif args.command == 'show-chain':
        show_chain()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()