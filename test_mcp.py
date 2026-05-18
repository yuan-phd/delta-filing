"""
Delta Filing — MCP Test Client
================================
Tests the EDGAR MCP server by connecting to it and calling tools.

This simulates what LangGraph will do later:
  1. Launch the MCP server as a subprocess
  2. Ask it "what tools do you have?"
  3. Call a tool and get results

Usage:
    python test_mcp.py
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    print("=" * 60)
    print("  Delta Filing — MCP Server Test")
    print("=" * 60)

    # Connect to the MCP server by launching it as a subprocess
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_edgar.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # --- Test 1: List available tools ---
            print("\n[1] Discovering tools...")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"    Tool: {tool.name}")
                print(f"      {tool.description[:70]}...")

            # --- Test 2: Call search_filings ---
            print("\n[2] Calling search_filings(ticker='AAPL')...")
            result = await session.call_tool(
                "search_filings",
                {"ticker": "AAPL", "filing_type": "10-K", "count": 3},
            )
            data = json.loads(result.content[0].text)
            for f in data:
                print(f"    {f['filing_date']} | {f['form_type']}")

            # --- Test 3: Call get_filing_section ---
            print("\n[3] Calling get_filing_section(ticker='AAPL', section='1A')...")
            result = await session.call_tool(
                "get_filing_section",
                {"ticker": "AAPL", "section_id": "1A"},
            )
            data = json.loads(result.content[0].text)
            print(f"    Section: {data['section_name']}")
            print(f"    Date: {data['filing_date']}")
            print(f"    Length: {data['text_length']} chars")
            print(f"    Preview: {data['text'][:150]}...")

            # --- Test 4: Call diff_filing_sections ---
            print("\n[4] Calling diff_filing_sections(ticker='AAPL', section='1A')...")
            result = await session.call_tool(
                "diff_filing_sections",
                {"ticker": "AAPL", "section_id": "1A"},
            )
            data = json.loads(result.content[0].text)
            s = data["summary"]
            print(f"    Comparing: {data['filing_a_date']} -> {data['filing_b_date']}")
            print(f"    Added: {s['added']} | Removed: {s['removed']} | "
                  f"Modified: {s['modified']} | Unchanged: {s['unchanged']}")

    print("\n" + "=" * 60)
    print("  MCP server test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
