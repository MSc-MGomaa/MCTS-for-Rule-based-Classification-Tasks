from Tree import Node
from ChildrenND import ChildrenND
from tqdm import tqdm
from Extent import Calculate
from Evaluation import Evaluation
import random
from Hashmap import my_dictionary
from ChildrenCD import ChildrenCD


class MCTS:
    def __init__(self, trainingSet, numIteration, minSupport, mValue):
        self.df = trainingSet
        self.numIterations = numIteration
        self.minSupport = minSupport
        self.mValue = mValue
        self.Root = self.Root_Pattern()
        self.dict_obj = my_dictionary()
        self.Already_Expanded_Patterns = []
        self.M = []
        self.P = []

    def Root_Pattern(self):
        Root_node = []
        for i in range(len(self.df.columns) - 1):
            column = self.df.iloc[:, i].tolist()
            interval = [min(sorted(column)), max(sorted(column))]
            Root_node.append(interval)
        return Root_node

    def Generate_ChildrenND(self, node):
        obj = ChildrenND(dataset=self.df, Pattern=node.pattern, Root=self.Root, minimum_support=self.minSupport)
        children = obj.Direct_ChildrenND()
        node.children.clear()
        for child in children:
            node.children.append(Node(child))

    def Generate_ChildrenND_Simulation(self, node):
        obj = ChildrenND(dataset=self.df, Pattern=node.pattern, Root=self.Root, minimum_support=self.minSupport)
        children = obj.Direct_ChildrenND_Simulation()
        node.children.clear()
        for child in children:
            node.children.append(Node(child))

    # Generate Children for Categorical CD:
    def Generate_ChildrenCD(self, node):
        obj = ChildrenCD(dataset=self.df, pattern=node.pattern, min_support=self.minSupport, Root=self.Root)
        children = obj.Direct_ChildrenCD()
        node.children.clear()
        for child in children:
            node.children.append(Node(child))

    def Generate_ChildrenCD_Simulation(self, node):
        obj = ChildrenCD(dataset=self.df, pattern=node.pattern, min_support=self.minSupport, Root=self.Root)
        children = obj.Direct_ChildrenCD_Simulation()
        node.children.clear()
        for child in children:
            node.children.append(Node(child))

    def TerminalND(self, node):
        obj = Calculate(dataset=self.df, pattern=node.pattern, root=self.Root, mValue=None, label=None)
        if obj.extentND()[1] < self.minSupport:
            return True
        else:
            return False

    def TerminalCD(self, node):
        obj = Calculate(dataset=self.df, pattern=node.pattern, root=self.Root, label=None, mValue=None)
        if obj.extentCD()[1] < self.minSupport:
            return True
        else:
            return False

    def Best_Child(self, node):
        evaluation_list = []
        for child in node.children:
            obj = Evaluation(child.reward_list, node.number_of_visits, child.number_of_visits)
            evaluation_list.append(obj.ucbTuned())
        bestC = evaluation_list.index(max(evaluation_list))
        best_Child = node.children[bestC]

        return best_Child

    def nodeSelectionND(self, node):
        while not self.TerminalND(node):
            if list(set(node.children) - set(node.already_expanded_children)):
                break
            else:
                # in case the all children are already expanded:
                node = self.Best_Child(node)
                self.Generate_ChildrenND(node)

        return node

    def Node_SelectionCD(self, node):
        while not self.TerminalCD(node):
            if list(set(node.children) - set(node.already_expanded_children)):
                break
            else:
                # in case the all children are already expanded:
                node = self.Best_Child(node)
                self.Generate_ChildrenCD(node)

        return node

    def Expand(self, node):
        Not_Expanded = list(set(node.children) - set(node.already_expanded_children))
        random_index = random.randint(0, len(Not_Expanded) - 1)
        New_Expanded_node = Not_Expanded[random_index]

        if New_Expanded_node.pattern in self.dict_obj.values():
            node.already_expanded_children.append(New_Expanded_node)
            position = list(self.dict_obj.values()).index(New_Expanded_node.pattern)
            New_Expanded_node = list(self.dict_obj.keys())[position]
            New_Expanded_node.parent.append(node)
            node = New_Expanded_node

        else:
            New_Expanded_node.parent.append(node)
            node.already_expanded_children.append(New_Expanded_node)
            self.dict_obj.add(New_Expanded_node, New_Expanded_node.pattern)
            node = New_Expanded_node

        return node

    def RolloutND(self, node, label):
        path = []
        path_of_patterns = []
        reward = 0
        while not self.TerminalND(node):
            self.Generate_ChildrenND_Simulation(node)
            if len(node.children) == 0:
                break
            else:
                random_index = random.randint(0, len(node.children) - 1)

                Selected_Child = node.children[random_index]
                obj = Calculate(dataset=self.df, pattern=Selected_Child.pattern, root=self.Root, mValue=self.mValue,
                                label=label)
                path.append(float(obj.mEstimateND()))
                path_of_patterns.append(Selected_Child.pattern)
                reward = max(path)
                node = Selected_Child
        # --------------------------------------------------------------------------------------
        # to implement the top-k-memory strategy where k in our example is 1
        self.M.append(path_of_patterns[path.index(reward)])

        return reward

    def RolloutCD(self, node, label):
        path = []
        path_of_patterns = []
        reward = 0
        while not self.TerminalCD(node):
            self.Generate_ChildrenCD_Simulation(node)
            random_index = random.randint(0, len(node.children) - 1)
            Selected_Child = node.children[random_index]
            obj2 = Calculate(dataset=self.df, root=self.Root, pattern=Selected_Child.pattern, label=label,
                             mValue=self.mValue)
            path.append(float(obj2.mEstimateCD()))
            path_of_patterns.append(Selected_Child.pattern)
            reward = max(path)
            node = Selected_Child
        # --------------------------------------------------------------------------------------
        # to implement the top-k-memory strategy where k in our example is 1
        self.M.append(path_of_patterns[path.index(reward)])

        return reward

    def Has_parent(self, node):
        if node.parent is []:
            return False
        else:
            return True

    def Update(self, node, reward):
        node.reward_list.append(reward)
        node.number_of_visits += 1
        Parents_List = []
        if self.Has_parent(node):
            [Parents_List.append(parent) for parent in node.parent]
        if len(Parents_List) > 0:
            for element in Parents_List:
                element.reward_list.append(reward)
                element.number_of_visits += 1
                if self.Has_parent(element):
                    for parent in element.parent:
                        Parents_List.append(parent)

    def Union(self, lst1, lst2):
        final_list = lst1 + lst2
        return final_list

    def monteCarloND(self):
        Labels = list(set(self.df.iloc[:, -1].tolist()))
        rootNode = Node(self.Root)
        Pool_Sets = []

        count = 0
        for Label in Labels:
            self.Generate_ChildrenND(rootNode)

            for i in tqdm(range(self.numIterations),
                          desc=f'Class "{Label}": {Labels.index(Label) + 1} Of {len(Labels)}', ascii=False, ncols=75):
                selectedNode = self.nodeSelectionND(rootNode)
                New_Expanded_Node = self.Expand(selectedNode)
                self.Already_Expanded_Patterns.append(New_Expanded_Node.pattern)
                Reward = self.RolloutND(New_Expanded_Node, Label)
                self.Update(New_Expanded_Node, Reward)

            self.P = self.Union(self.Already_Expanded_Patterns, self.M)

            # make sure there are not redundant patterns:
            Without_duplicates = []
            for elem in self.P:
                if elem not in Without_duplicates:
                    Without_duplicates.append(elem)

            # Append
            count += len(Without_duplicates)
            print(len(Without_duplicates))
            Pool_Sets.append([Label, Without_duplicates, self.Root])

            # reset the configurations for a new iteration:
            self.Already_Expanded_Patterns.clear()
            self.P.clear()
            self.M.clear()
            self.dict_obj.clear()

        print(f"The mean of all rules generated = {count / len(Labels)}")
        return Pool_Sets, count / len(Labels)

    def monteCarloCD(self):
        # for each class, we will retrieve a pool of patterns:
        Pool_Sets = []
        Labels = list(set(self.df.iloc[:, -1].tolist()))
        Root_Node = Node(self.Root)

        for Label in Labels:
            self.Generate_ChildrenCD(Root_Node)
            for i in tqdm(range(self.numIterations),
                          desc=f'Class "{Label}": {Labels.index(Label) + 1} Of {len(Labels)}', ascii=False, ncols=75):
                Selected_Node = self.Node_SelectionCD(Root_Node)
                New_Expanded_Node = self.Expand(Selected_Node)
                self.Already_Expanded_Patterns.append(New_Expanded_Node.pattern)
                Reward = self.RolloutCD(New_Expanded_Node, Label)
                self.Update(New_Expanded_Node, Reward)

            self.P = self.Union(self.Already_Expanded_Patterns, self.M)
            # make sure there are not redundant patterns:
            Without_duplicates = []
            for elem in self.P:
                if elem not in Without_duplicates:
                    Without_duplicates.append(elem)

            # Append
            print(len(Without_duplicates))
            Pool_Sets.append([Label, Without_duplicates, self.Root])

            # reset the configurations for a new iteration:
            self.Already_Expanded_Patterns.clear()
            self.P.clear()
            self.M.clear()
            self.dict_obj.clear()

        return Pool_Sets
